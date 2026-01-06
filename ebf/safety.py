import torch
import torch.nn as nn

def check_kernel_compatibility(module: nn.Module) -> bool:
    """
    Red Team Check: Does this module support bias modification?
    
    Returns:
        False if module is Quantized (BNB), Fused, or LoRA-wrapped.
        True if standard nn.Linear.
    """
    class_name = module.__class__.__name__
    
    # 1. Check for BitsAndBytes (Quantization)
    # BNB kernels often ignore Python bias attributes if initialized without them
    if "Linear8bitLt" in class_name or "Linear4bit" in class_name:
        return False
        
    # 2. Check for LoRA (PEFT)
    # Folding into a LoRA-wrapped layer is mathematically ambiguous 
    # (does it apply before or after the adapter?)
    if hasattr(module, "lora_A"):
        return False
        
    return True

def monitor_drift(original_logits, folded_logits, threshold=0.1):
    """
    Sanity check for semantic collapse.
    """
    probs_p = torch.nn.functional.softmax(original_logits, dim=-1)
    probs_q = torch.nn.functional.softmax(folded_logits, dim=-1)
    
    # KL Divergence
    kl_div = torch.sum(probs_p * (torch.log(probs_p) - torch.log(probs_q)), dim=-1).mean()
    
    if kl_div > threshold:
        return False # Drift detected
    return True