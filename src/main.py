import numpy as np
import matplotlib.pyplot as plt
from ai_essentials.mlp import MLP
from ai_essentials.optimizer import SGD
from ai_essentials.loss import mse_loss

# Generate synthetic nonlinear data: y = sin(3x) + 0.3x^2 - 0.5x
np.random.seed(42)
x_raw = np.linspace(-5, 5, 200)
y_raw = np.sin(3 * x_raw) + 0.3 * x_raw**2 - 0.5 * x_raw + np.random.normal(0, 0.2, size=x_raw.shape)

# Normalize inputs (zero mean, unit variance) — keeps pre-activations in tanh's sweet spot
x_mean, x_std = x_raw.mean(), x_raw.std()
x_norm = (x_raw - x_mean) / x_std

# Normalize targets — output layer can start near the right range immediately
y_mean, y_std = y_raw.mean(), y_raw.std()
y_norm = (y_raw - y_mean) / y_std

# Prepare data for the MLP (expects list of lists for x)
X = [[float(xi)] for xi in x_norm]
y_targets = [float(yi) for yi in y_norm]

model = MLP(
    num_inputs=1, layers=[(32, 'tanh'), (32, 'tanh'), (1, None)]
)  # tanh throughout preserves signal; linear output for regression
optimizer = SGD(model.parameters(), lr=0.05)

# Training loop with live plotting
plt.ion()
fig, ax = plt.subplots()
for epoch in range(1, 501):
    # Forward pass
    y_pred = [model(xi)[0] for xi in X]
    loss = mse_loss(y_pred, y_targets)
    # Backward pass and optimization step
    loss.backward()
    optimizer.clip_grad(5.0)  # prevent exploding gradients
    optimizer.step()
    optimizer.zero_grad()

    # Plot every 25 epochs (denormalize predictions back to original scale)
    if epoch % 25 == 0 or epoch == 1:
        ax.clear()
        ax.scatter(x_raw, y_raw, color="blue", s=5, label="True")
        y_plot = [yp.data * y_std + y_mean for yp in y_pred]
        ax.plot(x_raw, y_plot, color="red", linewidth=2, label="Predicted")
        ax.set_title(f"Epoch {epoch} | Loss (normalized): {loss.data:.4f}")
        ax.legend()
        plt.pause(0.1)

plt.ioff()
plt.show()

print("Training complete.")
