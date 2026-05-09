"""
backprop_from_scratch.py

multi-layer perceptron with full backpropagation implemented from scratch.
no pytorch, no tensorflow — only numpy.

supports:
  - arbitrary layer depth and width
  - activations: relu, sigmoid, tanh, leaky_relu, softmax
  - losses: cross-entropy, mse
  - optimizers: sgd, sgd+momentum, rmsprop, adam
  - regularization: l2, dropout
  - batch normalization (forward + backward)
  - gradient clipping
  - learning rate schedulers
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field


# ─────────────────────────── activations ───────────────────────────

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def relu_backward(dA: np.ndarray, z: np.ndarray) -> np.ndarray:
    return dA * (z > 0).astype(float)

def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(z > 0, z, alpha * z)

def leaky_relu_backward(dA: np.ndarray, z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    dz = np.ones_like(z)
    dz[z < 0] = alpha
    return dA * dz

def sigmoid(z: np.ndarray) -> np.ndarray:
    # numerically stable
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))

def sigmoid_backward(dA: np.ndarray, z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return dA * s * (1 - s)

def tanh_act(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

def tanh_backward(dA: np.ndarray, z: np.ndarray) -> np.ndarray:
    return dA * (1 - np.tanh(z) ** 2)

def softmax(z: np.ndarray) -> np.ndarray:
    # subtract max for numerical stability
    shifted = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(shifted)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


ACTIVATIONS = {
    "relu":       (relu,      relu_backward),
    "leaky_relu": (leaky_relu, leaky_relu_backward),
    "sigmoid":    (sigmoid,   sigmoid_backward),
    "tanh":       (tanh_act,  tanh_backward),
    "softmax":    (softmax,   None),   # softmax gradient merged with cross-entropy
}


# ─────────────────────────── losses ────────────────────────────────

def cross_entropy_loss(AL: np.ndarray, Y: np.ndarray, eps: float = 1e-12) -> float:
    m = Y.shape[1]
    return -np.sum(Y * np.log(AL + eps)) / m

def cross_entropy_backward(AL: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # combined softmax + cross-entropy gradient: dZ = AL - Y
    return AL - Y

def mse_loss(AL: np.ndarray, Y: np.ndarray) -> float:
    m = Y.shape[1]
    return np.sum((AL - Y) ** 2) / (2 * m)

def mse_backward(AL: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return (AL - Y) / Y.shape[1]


# ─────────────────────────── batch norm layer ──────────────────────

class BatchNorm:
    def __init__(self, n_features: int, momentum: float = 0.9, eps: float = 1e-8):
        self.gamma = np.ones((n_features, 1))
        self.beta  = np.zeros((n_features, 1))
        self.momentum = momentum
        self.eps = eps
        # running stats for inference
        self.running_mean = np.zeros((n_features, 1))
        self.running_var  = np.ones((n_features, 1))
        # cache for backward
        self._cache: Dict = {}

    def forward(self, Z: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            mu  = np.mean(Z, axis=1, keepdims=True)
            var = np.var(Z, axis=1, keepdims=True)
            Z_norm = (Z - mu) / np.sqrt(var + self.eps)
            self._cache = {"Z": Z, "Z_norm": Z_norm, "mu": mu, "var": var}
            # update running stats
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var  = self.momentum * self.running_var  + (1 - self.momentum) * var
        else:
            Z_norm = (Z - self.running_mean) / np.sqrt(self.running_var + self.eps)
        return self.gamma * Z_norm + self.beta

    def backward(self, dout: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        Z, Z_norm, mu, var = (self._cache[k] for k in ("Z", "Z_norm", "mu", "var"))
        m = Z.shape[1]
        dgamma = np.sum(dout * Z_norm, axis=1, keepdims=True)
        dbeta  = np.sum(dout, axis=1, keepdims=True)

        dZ_norm = dout * self.gamma
        dvar  = np.sum(dZ_norm * (Z - mu) * -0.5 * (var + self.eps) ** -1.5, axis=1, keepdims=True)
        dmu   = np.sum(dZ_norm * -1 / np.sqrt(var + self.eps), axis=1, keepdims=True) \
                + dvar * np.mean(-2 * (Z - mu), axis=1, keepdims=True)
        dZ    = dZ_norm / np.sqrt(var + self.eps) + dvar * 2 * (Z - mu) / m + dmu / m

        return dZ, dgamma, dbeta


# ─────────────────────────── layer config ──────────────────────────

@dataclass
class LayerConfig:
    n_units:    int
    activation: str = "relu"
    use_bn:     bool = False
    dropout_rate: float = 0.0   # 0 = disabled


# ─────────────────────────── mlp ───────────────────────────────────

class MLP:
    """
    fully connected neural network with arbitrary depth.
    supports batch norm, dropout, l2, and multiple optimizers.
    """

    def __init__(
        self,
        layer_configs: List[LayerConfig],
        loss: str = "cross_entropy",    # "cross_entropy" | "mse"
        optimizer: str = "adam",        # "sgd" | "momentum" | "rmsprop" | "adam"
        lr: float = 1e-3,
        l2_lambda: float = 0.0,
        clip_grad: Optional[float] = None,
        seed: int = 42,
    ):
        np.random.seed(seed)
        self.configs  = layer_configs
        self.L        = len(layer_configs)
        self.loss_fn  = loss
        self.opt_name = optimizer
        self.lr       = lr
        self.l2       = l2_lambda
        self.clip     = clip_grad

        self.params: Dict[str, np.ndarray] = {}
        self.bn_layers: Dict[int, BatchNorm] = {}
        self._cache: Dict = {}
        self._opt_state: Dict = {}
        self.t = 0   # adam timestep

        self._history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    # ── weight init ──────────────────────────────────────────────

    def init_params(self, n_input: int):
        dims = [n_input] + [cfg.n_units for cfg in self.configs]
        for l in range(1, self.L + 1):
            fan_in, fan_out = dims[l-1], dims[l]
            act = self.configs[l-1].activation

            # he init for relu variants, xavier for others
            if act in ("relu", "leaky_relu"):
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(2.0 / (fan_in + fan_out))

            self.params[f"W{l}"] = np.random.randn(fan_out, fan_in) * scale
            self.params[f"b{l}"] = np.zeros((fan_out, 1))

            if self.configs[l-1].use_bn:
                self.bn_layers[l] = BatchNorm(fan_out)

        # adam / rmsprop / momentum state
        for key, val in self.params.items():
            self._opt_state[f"v_{key}"] = np.zeros_like(val)
            self._opt_state[f"s_{key}"] = np.zeros_like(val)

    # ── forward ──────────────────────────────────────────────────

    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        A = X
        self._cache = {}
        dropout_masks: Dict[int, np.ndarray] = {}

        for l in range(1, self.L + 1):
            W = self.params[f"W{l}"]
            b = self.params[f"b{l}"]
            cfg = self.configs[l-1]

            Z = W @ A + b

            # batch norm before activation
            if cfg.use_bn and l in self.bn_layers:
                Z = self.bn_layers[l].forward(Z, training)

            act_fn, _ = ACTIVATIONS[cfg.activation]
            A_prev = A
            A = act_fn(Z)

            # dropout
            if training and cfg.dropout_rate > 0:
                mask = (np.random.rand(*A.shape) > cfg.dropout_rate).astype(float)
                mask /= (1 - cfg.dropout_rate)   # inverted dropout
                A = A * mask
                dropout_masks[l] = mask

            self._cache[f"A{l-1}"] = A_prev
            self._cache[f"Z{l}"]   = Z

        self._cache["dropout_masks"] = dropout_masks
        return A   # AL

    # ── backward ─────────────────────────────────────────────────

    def backward(self, AL: np.ndarray, Y: np.ndarray) -> Dict[str, np.ndarray]:
        grads: Dict[str, np.ndarray] = {}
        m = Y.shape[1]
        dropout_masks = self._cache["dropout_masks"]

        # output layer gradient
        cfg_out = self.configs[-1]
        if cfg_out.activation == "softmax" and self.loss_fn == "cross_entropy":
            dA = cross_entropy_backward(AL, Y)
        elif self.loss_fn == "mse":
            dA = mse_backward(AL, Y)
            _, back_fn = ACTIVATIONS[cfg_out.activation]
            dA = back_fn(dA, self._cache[f"Z{self.L}"])
        else:
            raise ValueError(f"unsupported loss/activation combo: {self.loss_fn}/{cfg_out.activation}")

        for l in reversed(range(1, self.L + 1)):
            cfg = self.configs[l-1]
            Z   = self._cache[f"Z{l}"]
            A_prev = self._cache[f"A{l-1}"]
            W   = self.params[f"W{l}"]

            # un-apply dropout
            if l in dropout_masks:
                dA = dA * dropout_masks[l]

            # gradient through activation (skip for output — already computed above)
            if l < self.L or not (cfg.activation == "softmax" and self.loss_fn == "cross_entropy"):
                _, back_fn = ACTIVATIONS[cfg.activation]
                if back_fn is not None:
                    dZ = back_fn(dA, Z)
                else:
                    dZ = dA   # softmax already handled
            else:
                dZ = dA

            # batch norm backward
            if cfg.use_bn and l in self.bn_layers:
                dZ, dgamma, dbeta = self.bn_layers[l].backward(dZ)
                grads[f"dgamma{l}"] = dgamma
                grads[f"dbeta{l}"]  = dbeta

            # parameter gradients + l2
            dW = (dZ @ A_prev.T) / m + (self.l2 / m) * W
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dA = W.T @ dZ

            grads[f"dW{l}"] = dW
            grads[f"db{l}"] = db

        # gradient clipping (global norm)
        if self.clip is not None:
            param_grads = [v for k, v in grads.items() if k.startswith("dW") or k.startswith("db")]
            global_norm = np.sqrt(sum(np.sum(g**2) for g in param_grads))
            if global_norm > self.clip:
                scale = self.clip / (global_norm + 1e-12)
                for k in grads:
                    grads[k] = grads[k] * scale

        return grads

    # ── optimizer step ───────────────────────────────────────────

    def update_params(self, grads: Dict[str, np.ndarray]):
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        keys = [f"W{l}" for l in range(1, self.L+1)] + [f"b{l}" for l in range(1, self.L+1)]
        # also update bn params if present
        for l in self.bn_layers:
            if f"dgamma{l}" in grads:
                keys += [f"gamma{l}", f"beta{l}"]

        for key in keys:
            grad_key = "d" + key
            if grad_key not in grads:
                continue

            g = grads[grad_key]
            v = self._opt_state[f"v_{key}"]
            s = self._opt_state[f"s_{key}"]

            if self.opt_name == "sgd":
                self.params[key] -= self.lr * g

            elif self.opt_name == "momentum":
                v = beta1 * v + (1 - beta1) * g
                self._opt_state[f"v_{key}"] = v
                self.params[key] -= self.lr * v

            elif self.opt_name == "rmsprop":
                s = beta2 * s + (1 - beta2) * g**2
                self._opt_state[f"s_{key}"] = s
                self.params[key] -= self.lr * g / (np.sqrt(s) + eps)

            elif self.opt_name == "adam":
                v = beta1 * v + (1 - beta1) * g
                s = beta2 * s + (1 - beta2) * g**2
                v_hat = v / (1 - beta1 ** self.t)
                s_hat = s / (1 - beta2 ** self.t)
                self._opt_state[f"v_{key}"] = v
                self._opt_state[f"s_{key}"] = s
                self.params[key] -= self.lr * v_hat / (np.sqrt(s_hat) + eps)

    # ── compute loss ─────────────────────────────────────────────

    def compute_loss(self, AL: np.ndarray, Y: np.ndarray) -> float:
        m = Y.shape[1]
        if self.loss_fn == "cross_entropy":
            loss = cross_entropy_loss(AL, Y)
        else:
            loss = mse_loss(AL, Y)
        # l2 penalty
        if self.l2 > 0:
            reg = sum(np.sum(self.params[f"W{l}"]**2) for l in range(1, self.L+1))
            loss += (self.l2 / (2 * m)) * reg
        return float(loss)

    # ── training loop ────────────────────────────────────────────

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
        lr_schedule: Optional[Callable[[int], float]] = None,
        verbose: int = 10,
    ):
        n_input = X_train.shape[0]
        self.init_params(n_input)
        m = X_train.shape[1]

        for epoch in range(1, epochs + 1):
            # lr schedule
            if lr_schedule is not None:
                self.lr = lr_schedule(epoch)

            # mini-batch shuffle
            perm = np.random.permutation(m)
            X_sh, Y_sh = X_train[:, perm], Y_train[:, perm]
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, m, batch_size):
                Xb = X_sh[:, start:start + batch_size]
                Yb = Y_sh[:, start:start + batch_size]

                AL    = self.forward(Xb, training=True)
                loss  = self.compute_loss(AL, Yb)
                grads = self.backward(AL, Yb)
                self.update_params(grads)

                epoch_loss += loss
                n_batches  += 1

            train_loss = epoch_loss / n_batches
            self._history["train_loss"].append(train_loss)

            if X_val is not None and Y_val is not None:
                AL_val   = self.forward(X_val, training=False)
                val_loss = self.compute_loss(AL_val, Y_val)
                self._history["val_loss"].append(val_loss)
                val_str = f"  val_loss={val_loss:.4f}"
            else:
                val_str = ""

            if verbose and epoch % verbose == 0:
                print(f"epoch {epoch:>4}/{epochs}  train_loss={train_loss:.4f}{val_str}  lr={self.lr:.6f}")

    # ── inference ────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X, training=False)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        AL = self.predict(X)
        if AL.shape[0] == 1:
            return (AL >= 0.5).astype(int).flatten()
        return np.argmax(AL, axis=0)

    def accuracy(self, X: np.ndarray, Y: np.ndarray) -> float:
        preds = self.predict_classes(X)
        if Y.shape[0] == 1:
            labels = Y.flatten().astype(int)
        else:
            labels = np.argmax(Y, axis=0)
        return float(np.mean(preds == labels))

    # ── plot ──────────────────────────────────────────────────────

    def plot_history(self):
        plt.figure(figsize=(9, 4))
        plt.plot(self._history["train_loss"], label="train loss", linewidth=2)
        if self._history["val_loss"]:
            plt.plot(self._history["val_loss"], label="val loss", linewidth=2, linestyle="--")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("training curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig("training_curve.png", dpi=120)
        plt.show()
        print("saved → training_curve.png")


# ─────────────────────────── gradient check ────────────────────────

def gradient_check(
    model: MLP,
    X: np.ndarray,
    Y: np.ndarray,
    eps: float = 1e-5,
    tolerance: float = 1e-4,
) -> bool:
    """
    numerically verifies backprop gradients via finite differences.
    only run on tiny models / small batches — O(params) forward passes.
    """
    print("running gradient check...")
    AL    = model.forward(X, training=False)
    grads = model.backward(AL, Y)

    errors = []
    for key in [f"W{l}" for l in range(1, model.L + 1)]:
        W    = model.params[key]
        dW   = grads[f"d{key}"]
        dW_num = np.zeros_like(W)

        it = np.nditer(W, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            orig = W[idx]

            W[idx] = orig + eps
            AL_p   = model.forward(X, training=False)
            loss_p = model.compute_loss(AL_p, Y)

            W[idx] = orig - eps
            AL_m   = model.forward(X, training=False)
            loss_m = model.compute_loss(AL_m, Y)

            W[idx] = orig
            dW_num[idx] = (loss_p - loss_m) / (2 * eps)
            it.iternext()

        diff  = np.linalg.norm(dW - dW_num) / (np.linalg.norm(dW) + np.linalg.norm(dW_num) + 1e-12)
        errors.append((key, diff))
        print(f"  {key}: relative diff = {diff:.2e}  {'✓' if diff < tolerance else '✗ FAIL'}")

    ok = all(d < tolerance for _, d in errors)
    print("gradient check", "PASSED ✓" if ok else "FAILED ✗")
    return ok


# ─────────────────────────── demo ──────────────────────────────────

if __name__ == "__main__":
    # synthetic spiral dataset (2-class)
    np.random.seed(0)
    N, D, C = 300, 2, 2

    def make_spiral(n, d, c, noise=0.1):
        X = np.zeros((n * c, d))
        Y = np.zeros(n * c, dtype=int)
        for j in range(c):
            ix = range(n * j, n * (j + 1))
            r  = np.linspace(0.0, 1, n)
            t  = np.linspace(j * 4, (j + 1) * 4, n) + np.random.randn(n) * noise
            X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
            Y[ix] = j
        return X.T, Y

    X, Y_flat = make_spiral(N, D, C)
    Y_oh = np.eye(C)[Y_flat].T   # one-hot (C, m)

    # lr cosine annealing
    def cosine_lr(epoch, warmup=10, max_lr=1e-3, min_lr=1e-5, total=150):
        if epoch <= warmup:
            return max_lr * epoch / warmup
        progress = (epoch - warmup) / (total - warmup)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * progress))

    net = MLP(
        layer_configs=[
            LayerConfig(64,  activation="relu",    use_bn=True,  dropout_rate=0.2),
            LayerConfig(64,  activation="relu",    use_bn=True,  dropout_rate=0.2),
            LayerConfig(32,  activation="leaky_relu"),
            LayerConfig(C,   activation="softmax"),
        ],
        loss="cross_entropy",
        optimizer="adam",
        lr=1e-3,
        l2_lambda=1e-4,
        clip_grad=5.0,
    )

    # gradient check on tiny subset before full training
    net.init_params(D)
    gradient_check(net, X[:, :8], Y_oh[:, :8])

    # full training
    net.fit(
        X, Y_oh,
        epochs=150,
        batch_size=64,
        lr_schedule=lambda e: cosine_lr(e, total=150),
        verbose=25,
    )

    acc = net.accuracy(X, Y_oh)
    print(f"\nfinal train accuracy: {acc:.4f}")
    net.plot_history()