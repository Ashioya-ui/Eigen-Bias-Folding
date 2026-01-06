Contributing to EBF

We welcome contributions, especially from those interested in Activation Engineering and Hardware-Aware ML.

Red Team Protocol

This repository operates under a strict "Red Team" mindset.

Do not break kernels. Any PR modifying model weights must pass check_kernel_compatibility.

Verify Linearity. If you change the injection point, you must provide a mathematical proof that the injection preserves linear mapping to the residual stream.

No Latency. Features that introduce inference-time overhead (hooks, extra layers) will be rejected. EBF is strictly $O(1)$.

Style

Type hints are mandatory.

Use black for formatting.

All functional changes require a unit test in tests/.