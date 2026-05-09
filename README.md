# Deep Learning Fundamentals

> Pure NumPy implementations of deep learning from scratch — no autograd frameworks, only hand-derived backpropagation.

This repository implements core deep learning primitives with full forward and backward passes, designed for learning the mechanics of neural networks.

**Stack:** Python 3.10+ · NumPy · Matplotlib (optional, for visualization)

---

## Structure

```
deep-learning-fundamentals/
├── README.md
└── dl-from-scratch/
    ├── backprop_from_scratch.py   # MLP engine + optimizers + gradient checking
    ├── cnn_from_scratch.py        # Conv2D (im2col) + pooling + CNN
    └── advanced_backprop.py       # ResNet · self-attention · LSTM · autograd tape
```

---

##  Quick Start

### Installation

```bash
pip install numpy matplotlib
```

### Run Examples

```bash
cd dl-from-scratch

# MLP with gradient checking on spiral dataset
python backprop_from_scratch.py

# CNN on synthetic 3-channel 16×16 data
python cnn_from_scratch.py

# ResNet block + attention + LSTM + autograd
python advanced_backprop.py
```

---

##  Modules

### `backprop_from_scratch.py`

Fully connected networks with arbitrary depth and width.

**Activations**

| Function   | Forward              | Backward                   |
| ---------- | -------------------- | -------------------------- |
| ReLU       | `max(0, z)`          | `dA * (z > 0)`             |
| Leaky ReLU | `z if z > 0 else αz` | `dA * (1 if z > 0 else α)` |
| Sigmoid    | `1 / (1 + e^-z)`     | `dA * σ(z) * (1 - σ(z))`   |
| Tanh       | `tanh(z)`            | `dA * (1 - tanh²(z))`      |
| Softmax    | `e^z / Σe^z`         | merged with cross-entropy  |

**Loss Functions**

- **Cross-entropy:** `-Σ y log(ŷ) / m`
- **MSE:** `Σ(ŷ - y)² / 2m`

Softmax + cross-entropy gradient is derived analytically as `∂L/∂z = ŷ - y`, bypassing the Jacobian.

**Batch Normalization** (full backward pass)

```
Forward:  μ = mean(Z),  σ² = var(Z)
          Z_norm = (Z - μ) / sqrt(σ² + ε)
          out = γ * Z_norm + β

Backward: dgamma = Σ dout * Z_norm
          dbeta  = Σ dout
          dZ_norm = dout * gamma
          dvar   = Σ dZ_norm * (Z - μ) * -0.5 * (σ² + ε)^-1.5
          dmu    = Σ dZ_norm / -sqrt(σ² + ε) + dvar * mean(-2(Z - μ))
          dZ     = dZ_norm / sqrt(σ² + ε) + dvar * 2(Z - μ)/m + dmu/m
```

**Dropout** (inverted dropout at train time)

```python
mask = (rand(*shape) > p) / (1 - p)   # scale at train, not at test
```

**Optimizers**

| Optimizer | Update Rule                                     |
| --------- | ----------------------------------------------- |
| SGD       | `W -= lr * dW`                                  |
| Momentum  | `v = β₁v + (1-β₁)dW;  W -= lr * v`              |
| RMSprop   | `s = β₂s + (1-β₂)dW²;  W -= lr * dW / (√s + ε)` |
| Adam      | bias-corrected m̂, v̂; `W -= lr * m̂ / (√v̂ + ε)`   |

**Gradient Clipping** (global L2 norm)

```python
global_norm = sqrt(Σ ||g||²)
if global_norm > clip: g *= clip / global_norm
```

**Gradient Checking** (numerical verification via finite differences)

```
dW_num[i,j] = (L(W + ε·eᵢⱼ) - L(W - ε·eᵢⱼ)) / 2ε
relative_diff = ||dW - dW_num|| / (||dW|| + ||dW_num||)
threshold: < 1e-4
```

**Learning Rate Schedule** (cosine annealing with linear warmup)

```
if epoch ≤ warmup:   lr = lr_max * epoch / warmup
else:                lr = lr_min + 0.5*(lr_max - lr_min) * (1 + cos(π * progress))
```

---

### `cnn_from_scratch.py`

Convolutional neural networks from scratch.

**Im2Col / Col2Im Transformation**

Converts every receptive field into a column, transforming convolution into matrix multiplication:

```
X:    (N, C, H, W)
col:  (N·out_h·out_w, C·k·k)    ← im2col
W:    (F, C·k·k)
out = col @ W.T  →  (N, F, out_h, out_w)

Backward: dW = dout_col.T @ col
          dX = col2im(dout_col @ W)
```

**Output Spatial Dimensions**

```
out_h = (H + 2p - k) // s + 1
out_w = (W + 2p - k) // s + 1
```

**Max Pooling Backward**

Gradient is routed only to the position of the maximum value (argmax mask saved during forward).

**Batch Norm 2D**

Reduction over axes (N, H, W) per channel.

---

### `advanced_backprop.py`

Advanced architectures and techniques.

#### Residual Block

```
y = relu(BN(conv2(relu(BN(conv1(x))))) + shortcut(x))

Backward:
  d_merged flows to both branches
  dX = d_main + d_shortcut          ← gradient highway
```

Projection shortcut (1×1 conv) applied when `in_ch ≠ out_ch` or `stride > 1`.

#### Scaled Dot-Product Self-Attention

```
Q = XWq,  K = XWk,  V = XWv
scores = QKᵀ / sqrt(d_k)
attn   = softmax(scores)
ctx    = attn @ V
out    = ctx Wo
```

**Softmax Backward** (stable form, avoids full Jacobian):

```
dS = softmax(S) * (dA - Σ(dA * softmax(S)))
```

**Gradient Flow:**

```
d_ctx    → d_attn = d_ctx @ Vᵀ,     d_V = attnᵀ @ d_ctx
d_scores → d_Q = d_scores @ K,      d_K = d_scoresᵀ @ Q
dX       = d_Q @ Wqᵀ + d_K @ Wkᵀ + d_V @ Wvᵀ
```

#### LSTM Cell (BPTT)

All 4 gates packed in one weight matrix for efficiency:

```
z = [h_{t-1}; x_t]
[i, f, g, o] = split(W @ z + b, 4)

i = σ(·),  f = σ(·),  g = tanh(·),  o = σ(·)
c_t = f * c_{t-1} + i * g
h_t = o * tanh(c_t)
```

**Backward Pass:**

```
do       = dh * tanh(c_t)
dc_t     = dh * o * (1 - tanh²(c_t)) + dc_next
df       = dc_t * c_{t-1}
di       = dc_t * g
dg       = dc_t * i
dc_{t-1} = dc_t * f

di_raw = di * i*(1-i)    ← sigmoid derivative
df_raw = df * f*(1-f)
dg_raw = dg * (1-g²)     ← tanh derivative
do_raw = do * o*(1-o)

dW += dgates @ zᵀ / m
dz  = Wᵀ @ dgates  →  dh_prev, dx_t
```

#### Gradient Tape (Autograd)

Lightweight autograd via Python closures — same computational graph approach as PyTorch/JAX:

```python
loss = (X.matmul(W1).relu().matmul(W2) - Y).pow(2).mean()
loss.backward()   # populates W1.grad, W2.grad
```

Each operation records a `_backward_fn` closure that calls `.backward()` on its inputs, propagating gradients through the graph.

---

##  Learning Concepts Covered

- ✅ Forward and backward propagation
- ✅ Activation functions and their derivatives
- ✅ Batch normalization
- ✅ Dropout regularization
- ✅ Gradient checking and validation
- ✅ Optimization algorithms (SGD, Momentum, Adam)
- ✅ Learning rate scheduling
- ✅ Convolutional layers (im2col transformation)
- ✅ Pooling operations
- ✅ Residual connections
- ✅ Self-attention mechanisms
- ✅ LSTM cells and backpropagation through time
- ✅ Computational graphs and automatic differentiation

---

##  References

- Goodfellow et al. — _Deep Learning_ (MIT Press, 2016)
- He et al. — _Deep Residual Learning for Image Recognition_ (CVPR 2016)
- Vaswani et al. — _Attention Is All You Need_ (NeurIPS 2017)
- Hochreiter & Schmidhuber — _Long Short-Term Memory_ (Neural Computation 1997)
- Stanford CS231n — Convolutional Neural Networks for Visual Recognition

---

##  Why From Scratch?

Understanding the low-level mechanics of deep learning is crucial for:

- Debugging neural networks effectively
- Designing custom architectures
- Optimizing performance-critical code
- Appreciating modern frameworks (PyTorch, TensorFlow, JAX)

This repository removes the black box and shows exactly what happens at every step.

---

##  License

This is educational code. Feel free to use, modify, and learn from it.
