Final Red Team Analysis: EBF V1.0

Date: Jan 6, 2026
Subject: Release Candidate Audit - The "Linearity" Pivot

1. The Critical Fix: Targeting Output Projections

During the final audit, we identified a theoretical flaw in steering the Gate Projection (Input of MLP).

Issue: The MLP contains non-linearities (SiLU). $SiLU(x+v) \neq SiLU(x) + v$. Steering the input creates unpredictable activation patterns that do not linearly map to "concepts" in the residual stream.

Resolution: V1.0 targets the Down Projection (Output of MLP).

Mathematical Proof: The Transformer block is $x_{l+1} = x_l + MLP(x_l)$.

$MLP(x_l) = W_{down}(\dots) + b_{down}$.

By adding $v$ to $b_{down}$, we get $x_{l+1} = x_l + MLP(x_l) + v$.

This is exact equivalence to adding a residual task vector.

2. Kernel Safety & Quantization

We verified that bitsandbytes (BNB) kernels for 4-bit quantization ignore the python bias attribute if the layer was initialized without one.

Guardrail: check_kernel_compatibility now strictly forbids folding into quantized modules.

Impact: EBF is currently restricted to FP16/BF16/FP32 models. This is an acceptable trade-off for correctness.

3. Persistence

Added save/load using standard Torch serialization. This decouples the "Adaptation State" from the "Model Weights," allowing instant context switching.

