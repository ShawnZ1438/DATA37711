import numpy as np
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


"""
For Random Kitchen Sinks (RKS)
"""
class RandomKitchenSinks:
    """
    Random Kitchen Sinks featurization with ridge regression.

    Features (assignment notation, h = D / 2 pairs):
        phi_sin_j(x) = sin(w_j^T x + b_j),  phi_cos_j(x) = cos(w_j^T x + b_j),
        b_j ~ U[0, 2*pi],  w_j ~ N_d(0, I_d / sigma^2)   (i.e. each entry of w_j has standard deviation 1 / sigma)
        z(x) = (phi_sin_1, phi_cos_1, ..., phi_sin_h, phi_cos_h) in R^{2h}; an intercept column of ones is prepended.
    Ridge regression: coef = argmin ||y - Phi c||^2 + lambda ||c[1:]||^2  (the intercept is not penalised),
    solved in closed form: (Phi^T Phi + lambda P) c = Phi^T y with P = diag(0, 1, ..., 1).
    """
    def __init__(self, D: int = 1000, sigma: float = 1.0, ridge_lambda: float = 1e-3, seed: int = 0):
        """
            D: total number of random features (= 2h; must be even because we use both sin and cos)
            sigma: scale parameter of the RBF kernel; w_j ~ N(0, I / sigma^2)
            ridge_lambda: ridge regularization strength
            seed: seed for drawing (w_j, b_j)
        """
        if D % 2 != 0:
            raise ValueError("D = 2h must be even (one sin and one cos feature per random direction)")
        self.D = D
        self.h = D // 2
        self.sigma = sigma
        self.ridge_lambda = ridge_lambda
        self.seed = seed
        self.W = None      # shape (d, h), drawn on the first call to _features (needs the input dimension d)
        self.b = None      # shape (h,)
        self.coef_ = None  # shape (D + 1,)

    def _draw(self, d: int):
        rng = np.random.default_rng(self.seed)
        self.W = rng.normal(0.0, 1.0 / self.sigma, size=(d, self.h))  # columns are w_1, ..., w_h
        self.b = rng.uniform(0.0, 2.0 * np.pi, size=self.h)

    def _features(self, X: np.ndarray) -> np.ndarray:
        """
        Map X (n, d) to the RKS features, shape (n, D + 1): an intercept column, then sin/cos pairs.
        The same (W, b) are reused for every later call (train, validation, test, plotting grid).
        """
        X = np.asarray(X, dtype=float)
        if self.W is None:
            self._draw(X.shape[1])
        proj = X @ self.W + self.b                      # (n, h): w_j^T x_i + b_j
        Z = np.empty((X.shape[0], self.D))
        Z[:, 0::2] = np.sin(proj)                       # phi_sin_j
        Z[:, 1::2] = np.cos(proj)                       # phi_cos_j
        return np.hstack([np.ones((X.shape[0], 1)), Z])

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the ridge regression model to the (1) features computed from the input data X and (2) target values y.
        """
        phi = self._features(X)  # shape (n, D + 1)
        penalty = self.ridge_lambda * np.eye(phi.shape[1])
        penalty[0, 0] = 0.0      # do not shrink the intercept
        self.coef_ = np.linalg.solve(phi.T @ phi + penalty, phi.T @ np.asarray(y, dtype=float))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the input data X.
        """
        phi = self._features(X)  # shape (n, D + 1)
        return phi @ self.coef_  # shape (n,)


def sweep_rks_model(sweep_D: List[int], X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
                    y_val: np.ndarray, sigma: float = 1.0, ridge_lambda: float = 1e-3, seed: int = 0) -> Tuple:
    """
    Sweep over the number of random features D (= 2h) with sigma and ridge_lambda fixed.
    Returns (best_model, best_val_mse, best_D, history), where history is a list of dicts with the train and
    validation MSE for every D (used for the two loss plots).
    """
    best_model = None
    best_val_mse = float('inf')
    best_D = None
    history = []
    for D in sweep_D:
        model = RandomKitchenSinks(D=D, sigma=sigma, ridge_lambda=ridge_lambda, seed=seed).fit(X_train, y_train)
        train_mse = mse(y_train, model.predict(X_train))
        val_mse = mse(y_val, model.predict(X_val))
        history.append({"D": D, "train_mse": train_mse, "val_mse": val_mse})
        if val_mse < best_val_mse:
            best_model = model
            best_val_mse = val_mse
            best_D = D

    return best_model, best_val_mse, best_D, history


"""
For 1-hidden-layer MLP (MLP)
"""
class MLP(nn.Module):
    """1-hidden-layer MLP for regression: x -> Linear(d_in, d_hidden) -> activation -> Linear(d_hidden, d_out)."""
    def __init__(self, d_in: int, d_hidden: int = 128, d_out: int = 1, lr: float = 1e-3, batch_size: int = 128,
                 activation: str = "relu", device: str = "cpu", num_epochs: int = 1000, seed: int = 0):
        """
            d_hidden: hidden dimension
            activation: activation function ("relu" or "tanh")
            lr: learning rate (Adam)
            batch_size: batch size
            num_epochs: number of passes over the training data
            seed: seed for the weight initialisation and the mini-batch shuffling
        """
        super().__init__()
        # DO NOT forget this -- GPU will make NN training much faster if you have access to one
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.seed = seed
        torch.manual_seed(seed)
        act = {"relu": nn.ReLU(), "tanh": nn.Tanh()}[activation]
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            act,
            nn.Linear(d_hidden, d_out),
        ).to(device)

    def fit(self, X: np.ndarray, y: np.ndarray):
        # turn the numpy data into torch tensors and move to the device
        X = torch.from_numpy(X).float().to(self.device)
        y = torch.from_numpy(y).float().view(-1, 1).to(self.device)

        self.net.train()  # make sure to set the model to training mode
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(TensorDataset(X, y), batch_size=self.batch_size, shuffle=True, generator=generator)
        for epoch in range(self.num_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(self.net(xb), yb)
                loss.backward()
                optimizer.step()
        self.net.eval()
        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            X = torch.from_numpy(X).float().to(self.device)
            return self.forward(X).detach().cpu().numpy().reshape(-1)


def mlp_sweep(sweep_values: List[int], X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray,
              y_val: np.ndarray, **fixed) -> Tuple:
    """
    Sweep over the hidden dimension d_hidden with every other hyper-parameter fixed (passed through **fixed,
    e.g. lr=1e-2, batch_size=64, num_epochs=1000). Returns (best_model, best_val_mse, best_d_hidden, history).
    """
    best_model = None
    best_val_mse = float('inf')
    best_h = None
    history = []
    for h in sweep_values:
        model = MLP(d_in=X_train.shape[1], d_hidden=h, **fixed).fit(X_train, y_train)
        train_mse = mse(y_train, model.predict(X_train))
        val_mse = mse(y_val, model.predict(X_val))
        history.append({"d_hidden": h, "train_mse": train_mse, "val_mse": val_mse})
        print(f"d_hidden={h:4d}  train MSE {train_mse:.4f}  val MSE {val_mse:.4f}")
        if val_mse < best_val_mse:
            best_model = model
            best_val_mse = val_mse
            best_h = h

    return best_model, best_val_mse, best_h, history
