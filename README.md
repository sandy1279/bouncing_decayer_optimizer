# Bouncing Decayer Optimizer

> A physics-inspired optimizer for PyTorch — adding decaying oscillations like a bouncing ball.

Gradient descent is great — but it can get stuck or converge too fast.  
This optimizer adds a **decaying sinusoidal perturbation**:
- Early → big "bounces" help explore.
- Later → oscillations fade, optimizer settles.

## 🧰 Usage SOON (Pypi verification pending)
```python
from bouncing_decayer_optimizer import BouncingDecayerOptimizer
import torch

model = torch.nn.Linear(10, 1)
optimizer = BouncingDecayerOptimizer(model.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 1)
    outputs = model(inputs)
    loss = torch.nn.functional.mse_loss(outputs, targets)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")
