import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm
import numpy as np
import os
from .safety import check_kernel_compatibility

class EBFAdapter:
    """
    Main controller for Eigen-Bias Folding.
    
    Architecture V3:
    Targets the DOWN_PROJ (Output) layer of MLPs to ensure linear independence 
    from the activation function (SiLU/GeLU).
    """
    
    def __init__(self, model_wrapper):
        self.wrapper = model_wrapper
        self.model = model_wrapper.model
        self.device = model_wrapper.device
        self.adapters: Dict[int, torch.Tensor] = {} # layer_idx -> vector
        self.original_biases: Dict[str, torch.Tensor] = {}
        
    def calibrate(self, task_acts: Dict[int, torch.Tensor], 
                  base_acts: Dict[int, torch.Tensor],
                  threshold: float = 2.5):
        """
        Runs Spectral Resonance Scoring (SRS) to identify task vectors.
        """
        print(f"[*] Starting Spectral Calibration (SRS) with threshold {threshold}...")
        
        for layer_idx in tqdm(task_acts.keys(), desc="Analyzing Layers"):
            t_act = task_acts[layer_idx].float()
            b_act = base_acts[layer_idx].float()
            
            # 1. Center Data
            base_mean = torch.mean(b_act, dim=0)
            centered_task = t_act - base_mean
            
            # 2. Spectral Analysis (SVD)
            try:
                # robust lowrank SVD
                _, S, V = torch.svd_lowrank(centered_task, q=1)
                top_singular_value = S[0].item()
                top_vector = V[:, 0]
            except RuntimeError:
                # Fallback for instability
                continue 

            # 3. Compute Resonance Score
            # Project base noise onto this vector to see how "loud" the background is
            base_proj = torch.matmul(b_act - base_mean, top_vector)
            base_noise_level = torch.std(base_proj).item() + 1e-6
            
            resonance_score = top_singular_value / base_noise_level
            
            if resonance_score > threshold:
                # 4. Orthogonal Rejection (Clean the vector)
                clean_vector = self._orthogonal_rejection(top_vector, b_act)
                self.adapters[layer_idx] = clean_vector

        print(f"[*] Calibration Complete. Identified {len(self.adapters)} resonant layers.")

    def _orthogonal_rejection(self, target_vec: torch.Tensor, base_data: torch.Tensor, k=10) -> torch.Tensor:
        """Removes directions corresponding to the top-k principal components of base data."""
        _, _, V_base = torch.svd_lowrank(base_data, q=k)
        
        rejection = torch.zeros_like(target_vec)
        for i in range(k):
            v_b = V_base[:, i]
            proj = torch.dot(target_vec, v_b) * v_b
            rejection += proj
            
        clean = target_vec - rejection
        return clean / (torch.norm(clean) + 1e-8)

    def fold_all(self, scale: float = 1.0):
        """
        Applies adapters to the DOWN_PROJ bias.
        """
        count = 0
        for layer_idx, vector in self.adapters.items():
            target_module = self.wrapper.get_mlp_output(layer_idx)
            
            # STRICT KERNEL CHECK
            if not check_kernel_compatibility(target_module):
                print(f"[WARN] Layer {layer_idx} incompatible (Quantized/Fused). Skipping.")
                continue

            self._fold_single_layer(layer_idx, target_module, vector, scale)
            count += 1
        print(f"[*] Folded biases into {count} layers (Output Projection).")

    def _fold_single_layer(self, layer_idx, module, vector, scale):
        # We target the Output Projection.
        # y = Wx + b. We want y' = y + v.
        # So y' = Wx + (b + v).
        
        # Ensure bias exists
        if module.bias is None:
            module.bias = nn.Parameter(torch.zeros(module.out_features, device=self.device, dtype=module.weight.dtype))
            self.original_biases[str(layer_idx)] = None
        else:
            self.original_biases[str(layer_idx)] = module.bias.data.clone()
            
        with torch.no_grad():
            v = vector.to(self.device).to(module.bias.dtype)
            # Direct addition, no matrix multiply needed for output bias injection
            module.bias.data.add_(v * scale)

    def save(self, path: str):
        """Saves the learned adapters (vectors) to disk."""
        save_dict = {
            "adapters": self.adapters,
            "version": "1.0"
        }
        torch.save(save_dict, path)
        print(f"[*] Saved adapter to {path}")

    def load(self, path: str):
        """Loads adapters from disk."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Adapter not found: {path}")
        data = torch.load(path, map_location=self.device)
        self.adapters = data["adapters"]
        print(f"[*] Loaded {len(self.adapters)} vectors.")

    def reset(self):
        """Unfolds biases, restoring original state."""
        for layer_idx, saved_bias in self.original_biases.items():
            module = self.wrapper.get_mlp_output(int(layer_idx))
            with torch.no_grad():
                if saved_bias is None:
                    module.bias = None
                else:
                    module.bias.data.copy_(saved_bias)
        self.original_biases.clear()
        print("[*] Model Reset.")

class ModelWrapper:
    """Abstracts architecture differences."""
    def __init__(self, model_name_or_obj):
        if isinstance(model_name_or_obj, str):
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_obj, device_map="auto")
        else:
            self.model = model_name_or_obj
        self.device = self.model.device

    def get_mlp_output(self, layer_idx: int) -> nn.Module:
        # Llama/Mistral: model.layers[i].mlp.down_proj
        # This is where the output of the MLP is projected back to residual dimension
        try:
            return self.model.model.layers[layer_idx].mlp.down_proj
        except AttributeError:
            # Fallback for simple structures
            return self.model.layers[layer_idx].mlp.down_proj