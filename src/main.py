import numpy as np
import matplotlib.pyplot as plt
from ai_essentials.mlp import MLP
from ai_essentials.optimizer import SGD
from ai_essentials.loss import mse_loss

# Generate synthetic nonlinear data: y = sin(3x) + 0.3x^2 - 0.5x
np.random.seed(42)
x = np.linspace(-5, 5, 200)
y_true = np.sin(3 * x) + 0.3 * x ** 2 - 0.5 * x + np.random.normal(0, 0.2, size=x.shape)

# Prepare data for the MLP (expects list of lists for x)
X = [[float(xi)] for xi in x]
y_targets = [float(yi) for yi in y_true]

model = MLP(num_inputs=1, layers=[16, 16, (1, None)])  # deeper net for harder function
optimizer = SGD(model.parameters(), lr=0.01)

# Training loop with live plotting
plt.ion()
fig, ax = plt.subplots()
for epoch in range(1, 201):
    # Forward pass
    y_pred = [model(xi)[0] for xi in X]
    loss = mse_loss(y_pred, y_targets)
    # Backward pass and optimization step
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Plot every 10 epochs
    if epoch % 10 == 0 or epoch == 1:
        ax.clear()
        ax.scatter(x, y_true, color='blue', label='True')
        ax.plot(x, [yp.data for yp in y_pred], color='red', label='Predicted')
        ax.set_title(f'Epoch {epoch} | Loss: {loss.data:.4f}')
        ax.legend()
        plt.pause(0.1)

plt.ioff()
plt.show()

print('Training complete.')

