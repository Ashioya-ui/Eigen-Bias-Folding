Eigen-Bias Folding (EBF)

"Adaptation is a state change, not a weight training loop."

Eigen-Bias Folding (EBF) is a production-grade library for Zero-Latency Model Adaptation. It mathematically "folds" task-specific steering vectors into the bias parameters of pre-trained LLMs output projections, allowing you to ship "fine-tuned" behavior without shipping fine-tuned weights or incurring inference overhead.

🌟 Core Innovations

Zero-Latency Folding ($O(1)$): We target the Output Projection (down_proj) of MLP blocks. By merging adaptation vectors into existing bias terms ($b' = b + v$), we achieve mathematically equivalent results to residual stream injection with zero additional FLOPs.

Spectral Resonance Scoring (SRS): EBF analyzes the Singular Value spectrum of every layer to identify where the "Task" physically resides in the network, folding only into high-resonance layers to avoid "butterfly effect" noise.

Kernel-Aware Safety: Built-in guardrails prevent silent failures on quantized (BitsAndBytes) or fused kernels, ensuring production reliability.

📦 Installation

git clone [https://github.com/Ashioya-ui/Eigen-Bias-Folding.git](https://github.com/Ashioya-ui/Eigen-Bias-Folding.git)
cd Eigen-Bias-Folding
pip install -e .


⚡ Workflow

from ebf import EBFAdapter, ModelWrapper
import torch

# 1. Wrap your model (Supports Llama, Mistral, Falcon)
model = ModelWrapper("meta-llama/Llama-2-7b-hf")
adapter = EBFAdapter(model)

# 2. Calibrate (Gradient-Free)
# SRS automatically finds the best layers and scales
adapter.calibrate(
    task_acts=legal_activations,
    base_acts=general_activations,
    threshold=2.5  # Only fold layers with >2.5x signal-to-noise ratio
)

# 3. Save the "Ghost" Adapter (KB, not GB)
adapter.save("legal_context_v1.ebf")

# 4. Deployment (Hard Fold)
# This modifies the model in RAM. Zero inference overhead.
adapter.load("legal_context_v1.ebf")
adapter.fold_all(scale=1.0) 

# Run inference...
model.generate("Explain the tort of negligence.")


📐 The Math (Why Output Projection?)

Most steering methods inject vectors at the input of a layer (gate_proj). However, the MLP block contains non-linearities ($\sigma$):
$$ y = W_{down}(\sigma(W_{gate}(x + v))) $$
This distorts the manifold non-linearly.

EBF targets the Output Projection:
$$ y = W_{down}(\dots) + b_{down} + v $$
This is mathematically identical to adding a residual vector ($x_{l+1} = x_l + MLP(x_l) + v$), but incurs no runtime cost because $v$ is pre-added to $b_{down}$.

🛡️ Safety & Audits

This library was built under strict Red Team protocols.

Kernel Safety: Detects quantization (BNB/GPTQ) and prevents folding if it would break fused kernels.

Persistence: Decouples adaptation state from model weights via .ebf artifacts.

See RED_TEAM_FINAL.md for details.