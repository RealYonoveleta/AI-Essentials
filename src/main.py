import random
import numpy as np
import matplotlib.pyplot as plt
from ai_essentials.mlp import MLP
from ai_essentials.optimizer import SGD
from ai_essentials.loss import mse_loss

# Generate synthetic nonlinear data: y = sin(3x) + 0.3x^2 - 0.5x
np.random.seed(42)
x_raw = np.linspace(-5, 5, 200)
y_raw = (
    np.sin(3 * x_raw)
    + 0.3 * x_raw**2
    - 0.5 * x_raw
    + np.random.normal(0, 0.2, size=x_raw.shape)
)

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
    num_inputs=1, layers=[(64, "tanh"), (64, "tanh"), (64, "tanh"), (1, None)]
)  # wider network = more basis functions to compose the sine oscillations
optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9)

# Mini-batch size: 16 samples cover ~1-2 oscillation periods locally, avoiding
# cross-period gradient cancellation that kills high-frequency learning
batch_size = 16
indices = list(range(len(X)))

# Training loop with live plotting
plt.ion()
fig, ax = plt.subplots()
for epoch in range(1, 1001):
    # LR step decay: fall off after the parabola shape is locked in
    if epoch == 400:
        optimizer.lr *= 0.3
    if epoch == 700:
        optimizer.lr *= 0.3

    # --- True SGD: shuffle and train on random mini-batches each epoch ---
    random.shuffle(indices)
    for start in range(0, len(X), batch_size):
        batch_idx = indices[start : start + batch_size]
        X_batch = [X[i] for i in batch_idx]
        y_batch = [y_targets[i] for i in batch_idx]

        y_pred_batch = [model(xi)[0] for xi in X_batch]
        loss = mse_loss(y_pred_batch, y_batch)
        loss.backward()
        optimizer.clip_grad(5.0)
        optimizer.step()
        optimizer.zero_grad()

    # Plot every 25 epochs — evaluate on full dataset for a clean loss reading
    if epoch % 25 == 0 or epoch == 1:
        y_pred_full = [model(xi)[0] for xi in X]
        full_loss = mse_loss(y_pred_full, y_targets)
        ax.clear()
        ax.scatter(x_raw, y_raw, color="blue", s=5, label="True")
        y_plot = [yp.data * y_std + y_mean for yp in y_pred_full]
        ax.plot(x_raw, y_plot, color="red", linewidth=2, label="Predicted")
        ax.set_title(f"Epoch {epoch} | Loss (normalized): {full_loss.data:.4f}")
        ax.legend()
        plt.pause(0.1)

plt.ioff()
plt.show()

print("Training complete.")
