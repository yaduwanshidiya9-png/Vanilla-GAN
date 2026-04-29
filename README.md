# Vanilla GAN — Face Generation on CelebA

A minimal **Vanilla GAN** (Generative Adversarial Network) implementation in **PyTorch**, trained on the **CelebA** dataset to generate 64×64 RGB face images. Built using simple fully-connected (MLP) layers — no convolutions — to demonstrate the core GAN idea introduced by Goodfellow et al. (2014).

> 📓 Entire implementation lives in a single notebook: [`vanilla_gan.ipynb`](./vanilla_gan.ipynb)

---

## Overview

A GAN consists of two networks playing a min-max game:

- **Generator (G):** Takes a random noise vector `z ∈ ℝ¹⁰⁰` and produces a fake 64×64 RGB image.
- **Discriminator (D):** Takes an image (real or fake) and predicts the probability of it being real.

Training alternates between:
1. Updating **D** to better distinguish real CelebA faces from generated ones.
2. Updating **G** to fool **D** into classifying its outputs as real.

---

## Architecture

### Generator (MLP)
```
z (100) 
  → Linear(256)  → ReLU
  → Linear(512)  → ReLU
  → Linear(1024) → ReLU
  → Linear(64·64·3) → Tanh
  → reshape → (3, 64, 64)
```

### Discriminator (MLP)
```
img (3, 64, 64) 
  → Flatten
  → Linear(1024) → LeakyReLU(0.2)
  → Linear(512)  → LeakyReLU(0.2)
  → Linear(256)  → LeakyReLU(0.2)
  → Linear(1)    → Sigmoid
```

---

## Repository Structure

```
Vanilla-GAN/
├── vanilla_gan.ipynb      # Full training notebook
├── requirements.txt      # Python dependencies
├── .gitignore
└── README.md
```

---

## Dataset — CelebA

This project uses the **CelebA (Aligned & Cropped)** dataset.

- Download: [CelebA on Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) or the [official site](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- File expected: `img_align_celeba.zip`

---

## Installation

```bash
git clone https://github.com/yaduwanshidiya9-png/Vanilla-GAN.git
cd Vanilla-GAN
pip install -r requirements.txt
```
---

## Usage

### Option 1 — Google Colab (recommended)
The notebook is Colab-ready. It mounts Google Drive and unzips the dataset:

```python
from google.colab import drive
drive.mount('/content/drive')

!unzip -q "/content/drive/MyDrive/GANs/img_align_celeba.zip" -d "/content/"
```

Then simply **Runtime → Run all**.

### Option 2 — Local machine
1. Download `img_align_celeba.zip` and extract it so the images sit in `./img_align_celeba/`.
2. Open the notebook:
   ```bash
   jupyter notebook vanilla_gan.ipynb
   ```
3. Skip the Drive-mount and unzip cells, then run the rest.

### Training
Default settings (defined inside the notebook):
- `batch_size = 128`
- `epochs = 25`
- `z_dim = 100`
- `lr = 0.0002`

Auto-detects device — **MPS (Apple Silicon) → CUDA → CPU**.

---

## Sampling Generated Images

After (or during) training:
```python
save_generated_images(generator, epoch, device, num_imgs=8)
```
Outputs a 4×2 grid of generated faces, normalized to `[0, 1]` for display.

---

## Reference

> Goodfellow, I. et al. (2014). *Generative Adversarial Networks*. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661)

---

**Diya Yaduwanshi** — [@yaduwanshidiya9-png](https://github.com/yaduwanshidiya9-png)
