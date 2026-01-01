# MCNet: Scalable Bid Landscape Forecasting in Real-time Bidding

This repository contains a **PyTorch implementation of MCNet (Mixture Density Censored Network)** proposed in the KDD 2019 paper: *"[Scalable Bid Landscape Forecasting in Real-time Bidding](https://dl.acm.org/doi/10.1145/3292500.3330836)"*.

MCNet is a deep learning-based model designed to forecast the distribution of winning prices (bid landscapes) in Real-Time Bidding (RTB) environments, specifically handling **censored data** (winning vs. losing logs) effectively.

## 📌 Background & Motivation

In RTB (Real-Time Bidding), predicting the **winning price** (the market price of an ad slot) is crucial for:
* **Bid Shading** in First-Price Auctions (finding the optimal gap between value and bid).
* **Budget Pacing** & Allocation.

However, standard regression models fail due to:
1.  **Censored Data:** We only observe the true winning price when we win. When we lose, we only know the lower bound (winning price > my bid).
2.  **Multi-modal Distributions:** Bid landscapes are often complex and multi-modal, which single-point estimates (MSE) or single Gaussian assumptions cannot capture.

**MCNet solves this by combining Deep Neural Networks with Mixture Density Networks (MDN) and a Censored NLL Loss.**

## 🚀 Features

* **Mixture Density Network (MDN):** Predicts parameters ($\pi, \mu, \sigma$) for $K$ Gaussian distributions to capture multi-modal landscapes.
* **Censored Regression Loss:** Implements a specialized Negative Log-Likelihood (NLL) loss that handles:
    * **Winning Logs:** Maximizes Probability Density Function (PDF).
    * **Losing Logs:** Maximizes Survival Probability (1 - CDF).
* **Scalability:** Based on standard Deep Learning architecture, capable of handling high-dimensional sparse features unlike tree-based methods.

## 🛠️ Requirements

* Python 3.x
* PyTorch
* NumPy

```bash
pip install torch numpy
```


## 📂 Code Structure

* `MCNet`: The main model class.
* Input: High-dimensional features ($x$)
* Output: Mixture parameters ($\pi, \sigma, \mu$)


* `censored_nll_loss`: The core loss function implementing Equation (6) from the paper.

## 💻 Usage

### 1. Model Initialization

```python
from mcnet import MCNet, censored_nll_loss
import torch

# Hyperparameters
INPUT_DIM = 20
HIDDEN_DIM = 64
K = 4  # Number of mixture components

model = MCNet(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_mixtures=K)
print(model)

```

### 2. Training Step (with Dummy Data)

The model requires pairs of `(target, is_win)` for training.

* `target`: True winning price (if win) OR My bid price (if lose).
* `is_win`: 1 if won, 0 if lost.

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Dummy Data
x_batch = torch.randn(1024, INPUT_DIM)
target_batch = torch.abs(torch.randn(1024, 1) * 100 + 200) # Price
is_win_batch = torch.randint(0, 2, (1024, 1)).float()      # Win(1)/Lose(0)

model.train()

# 1. Forward
pi, sigma, mu = model(x_batch)

# 2. Compute Loss (Censored NLL)
loss = censored_nll_loss(pi, sigma, mu, target_batch, is_win_batch)

# 3. Backward
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"Training Loss: {loss.item():.4f}")

```

## 🧠 Technical Details

### Loss Function (Censored NLL)

The loss function distinguishes between winning and losing auctions:

$$\mathcal{L} = - \sum_{i \in \text{Win}} \log \left( \sum_{k=1}^{K} \pi_k \mathcal{N}(z_i) \right) - \sum_{i \in \text{Lose}} \log \left( \sum_{k=1}^{K} \pi_k (1 - \Phi(z_i)) \right)$$

* **Win:** We maximize the likelihood of the observed price .
* **Lose:** We maximize the probability that the winning price is greater than our bid  (Survival Probability).

### Comparison with pCTR

This approach is analogous to **LogLoss (Binary Cross Entropy)** in pCTR modeling:

* pCTR:  $y \cdot \log(p) + (1-y) \cdot \log(1-p)$
* MCNet: $I_{win} \cdot \log(PDF) + I_{lose} \cdot \log(CDF_{survival})$

## 📝 Reference

> Ren, Kan, et al. "Scalable bid landscape forecasting in real-time bidding." Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2019.

---

*Implementation by Soobin.Choi*

```

```
