# Fashion DCGAN Generator 👗🧥👟
> **Deep Convolutional Generative Adversarial Network (DCGAN) implemented in PyTorch for high-fidelity synthetic fashion apparel synthesis, with an interactive Streamlit UI.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Project Overview

This repository features a complete, end-to-end implementation of a **Deep Convolutional Generative Adversarial Network (DCGAN)** trained on the **Fashion-MNIST** dataset. The model generates 28x28 grayscale clothing items (t-shirts, dresses, sneakers, coats, bags) from a 100-dimensional continuous latent space Gaussian distribution.

An interactive **Streamlit dashboard** (`app.py`) is bundled to enable real-time inference, seed-based image generation, latent space sampling, and model inspection without writing code.

---

## 🏛️ Network Architecture

### 1. Generator
The Generator transforms a 100-dimensional latent noise vector \( z \sim \mathcal{N}(0, 1) \) into a synthesized \( 28 	imes 28 \) image through fractional-strided convolutions:
- **Input:** Latent vector $(100, 1, 1)$
- **Layer 1:** `ConvTranspose2d(100, 128, kernel=7, stride=1, padding=0)` $ightarrow$ `BatchNorm2d` $ightarrow$ `ReLU` $ightarrow$ Output $(128, 7, 7)$
- **Layer 2:** `ConvTranspose2d(128, 64, kernel=4, stride=2, padding=1)` $ightarrow$ `BatchNorm2d` $ightarrow$ `ReLU` $ightarrow$ Output $(64, 14, 14)$
- **Layer 3:** `ConvTranspose2d(64, 1, kernel=4, stride=2, padding=1)` $ightarrow$ `Tanh` $ightarrow$ Output $(1, 28, 28)$

### 2. Discriminator
The Discriminator acts as a binary classifier evaluating whether an input image is real (from Fashion-MNIST) or generated:
- **Input:** Image $(1, 28, 28)$
- **Layer 1:** `Conv2d(1, 64, kernel=4, stride=2, padding=1)` $ightarrow$ `LeakyReLU(0.2)` $ightarrow$ `Dropout2d(0.3)`
- **Layer 2:** `Conv2d(64, 128, kernel=4, stride=2, padding=1)` $ightarrow$ `LeakyReLU(0.2)` $ightarrow$ `Dropout2d(0.3)`
- **Layer 3:** `Flatten` $ightarrow$ `Linear(128 * 7 * 7, 1)` $ightarrow$ `Sigmoid`

---

## ⚙️ Hyperparameters & Training Specs

| Parameter | Value | Description |
|---|---|---|
| **Latent Dimension ($z$)** | 100 | Standard Normal Distribution |
| **Batch Size** | 64 | Mini-batch training size |
| **Optimizer** | Adam | Both Generator & Discriminator |
| **Learning Rate ($lpha$)** | 0.0002 | Tuned for stable minimax convergence |
| **Betas $(eta_1, eta_2)$** | (0.5, 0.999) | Standard DCGAN momentum stabilization |
| **Loss Function** | Binary Cross Entropy (`nn.BCELoss`) | Minimax GAN Objective |

---

## 🚀 Quickstart & Installation

### 1. Clone the repository
```bash
git clone https://github.com/abhi7abhi/fashion-dcgan-generator.git
cd fashion-dcgan-generator
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train.py
```
*Note: Checkpoints and loss curves will be automatically logged during training.*

### 4. Launch the Interactive Streamlit Web App
```bash
streamlit run app.py
```
Navigate to `http://localhost:8501` to test latent space manipulation and generate custom fashion apparel on the fly!

---

## 📂 Project Structure

```
fashion-dcgan-generator/
├── app.py              # Streamlit interactive application
├── model.py            # Generator & Discriminator PyTorch architectures
├── train.py            # Training loop, data loaders, and checkpointing
├── requirements.txt    # Required Python libraries
└── README.md           # Documentation
```

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
