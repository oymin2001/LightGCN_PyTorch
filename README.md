---

# LightGCN with PyTorch 🚀

This repository implements the **Light Graph Convolutional Network (LightGCN)** for recommendation systems using PyTorch. The implementation is based on the [LightGCN paper](https://arxiv.org/abs/2002.02126) and focuses on **memory efficiency** and **scalability** for large-scale datasets.

---

## 🌟 Key Features
- **Chunk-wise DataLoader**: Efficiently computes the adjacency matrix for massive datasets.
- **Scalable Evaluation**: Metrics like **Recall**, **Precision**, **NDCG**, and **MAP** are calculated in chunks to handle large-scale data.
- **Performance Tracking**: Comprehensive metric evaluation for each training epoch.

---

## 📊 Results
Below is the model performance over **30 epochs**:

![Result Plot](https://github.com/user-attachments/assets/8ef2681b-2dd2-42e6-bf5d-2ffb75e15800)

- **Metrics Evaluated**:  
  - **Recall@20**  
  - **Precision@20**  
  - **NDCG@20**  
  - **MAP@20**  

The graph shows steady improvement, especially in Recall and NDCG, while other metrics follow a consistent upward trend.

---

## 🧩 Repository Structure
- `LightGCN.ipynb` – **Example notebook** to demonstrate model usage.
- `Gentle_Introduction_to_LightGCN.ipynb` – **Step-by-step guide** to understand LightGCN.
- `module/` – Core components for training, evaluation, and model definition.
- `dataset/` – Sample datasets for testing.


## 🚀 Quick Start
1. Prepare your dataset and configure the environment.
2. Run the provided notebooks:
   - Start with **`Gentle_Introduction_to_LightGCN.ipynb`** for the workflow overview.
   - Use **`LightGCN.ipynb`** for running training and evaluation.

---

## 🔗 References
- [Recommenders Repository](https://github.com/recommenders-team/recommenders)
- [LightGCN Paper](https://arxiv.org/abs/2002.02126)

---
