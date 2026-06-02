![](https://media.tenor.com/bhC8X-tsTK4AAAAi/tspchan1-lick.gif)

# MNIST Binary Classifier — From Scratch & with Burn via Rust

A binary neural network classifier for MNIST digits (0 vs. 1), implemented twice:

- **`NN_Rust_CPU/`** — Pure Rust, no ML framework, compiled directly with `rustc`
- **`NN_Rust_GPU/`** — Same architecture using the [Burn](https://burn.dev) deep learning framework with CUDA acceleration

---

## Architecture
```
Input (784)  →  Linear  →  ReLU  →  Linear  →  Sigmoid  →  Output (1)
[Hidden: 20]

| Parameter     | Value                |
|---------------|----------------------|
| Input size    | 784 (28×28 px)       |
| Hidden units  | 20                   |
| Output        | 1 (binary)           |
| Activation    | ReLU → Sigmoid       |
| Loss          | Binary Cross-Entropy |
| Optimizer     | SGD                  |
| Learning rate | 0.01                 |
| Batch size    | 16                   |
| Epochs        | 10                   |

---

```
## Getting Started

### Prerequisites

- [Rust](https://rustup.rs/) (stable)
- CUDA toolkit (for `burn_model` only)
- MNIST dataset (see below)

### Download MNIST Data

```bash
mkdir -p data/MNIST/raw && cd data/MNIST/raw

curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gunzip *.gz
```

### Run — Pure Rust (CPU)

```bat
cd NN_Rust_CPU
run.bat
```

Or manually:

```bat
rustc -O NN_model.rs && NN_model.exe
```

### Run — Burn + CUDA (GPU)

```bat
cd NN_Rust_GPU
run.bat
```

Or manually:

```bat
cargo build --release
cargo run --release
```
---

## Implementation Details

### `neuron/` — Pure Rust

All neural network primitives are implemented from scratch using a custom `Matrix2D` struct:

- **Forward pass** — manual matrix multiplication, bias addition, ReLU, Sigmoid
- **Backpropagation** — chain rule applied by hand through each layer
- **Gradient descent** — weights and biases updated directly after each batch
- **Data layout** — columns-as-samples convention (`[features × samples]`)

Key types:
```
Matrix2D      — 2D tensor with row-major storage
LinearLayer   — weights + biases, forward/backward methods
Neuron        — two-layer network with training loop
LayerCache    — stores activations for backprop
```

### `burn_model/` — Burn Framework

Uses Burn's autodiff engine on top of a CUDA backend for GPU-accelerated training:

- **Backend** — `Autodiff<Cuda>`
- **Autograd** — gradients computed automatically via `loss.backward()`
- **Optimizer** — `SgdConfig` from `burn::optim`
- **Data layout** — rows-as-samples convention (`[samples × features]`)

---
## Key Differences Between Implementations

| Aspect          | `neuron/` (Pure Rust)    | `burn_model/` (Burn)          |
|-----------------|--------------------------|-------------------------------|
| Autograd        | Manual backprop          | Automatic differentiation     |
| Hardware        | CPU only                 | CUDA GPU                      |
| Matrix layout   | `[features × samples]`   | `[samples × features]`        |
| Weight init     | Custom PRNG (hash-based) | Burn default initializer      |
| Dependencies    | None (std only)          | `burn`, `burn-cuda`           |
| Purpose         | Learning / from-scratch  | Production-style training     |

---

## Dependencies

### `neuron/`
```toml
[dependencies]
# none — pure std
```

### `burn_model/`
```toml
[dependencies]
burn = { version = "0.16", features = ["autodiff"] }
burn-cuda = "0.16"
```

---

## License

MIT


