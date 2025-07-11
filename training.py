import numpy as np

class TinyOnlineLearner:
    def __init__(self, input_dim, output_dim, lr=0.001):
        # Small single-layer net: input → output
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros((1, output_dim))
        self.lr = lr

    def predict(self, x):
        # x shape: (1, input_dim)
        return x @ self.W + self.b

    def train_one(self, x, y):
        # Forward pass
        y_pred = self.predict(x)
        # Simple squared error loss
        loss = ((y_pred - y) ** 2).mean()
        # Compute gradient
        grad_W = x.T @ (y_pred - y) * (2 / x.shape[0])
        grad_b = np.mean(y_pred - y, axis=0, keepdims=True) * 2
        # Update weights (nudge a tiny bit)
        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b
        return loss

# Example usage
if __name__ == "__main__":
    np.random.seed(42)
    learner = TinyOnlineLearner(input_dim=3, output_dim=1, lr=0.01)

    # Simulated loop: new data every turn
    for step in range(50):
        x = np.random.rand(1, 3)            # Random input
        y = np.sum(x) + np.random.randn(1,1)*0.1  # True output: sum plus noise

        loss = learner.train_one(x, y)
        y_pred = learner.predict(x)

        print(f"Step {step+1:2d}: Loss={loss:.4f}, Pred={y_pred.ravel()[0]:.3f}, Target={y.ravel()[0]:.3f}")

    print("Final weights:", learner.W)
