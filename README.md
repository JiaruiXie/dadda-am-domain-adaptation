# Domain-Adversarial and Decision Distribution Alignment (DADDA)

A lightweight PyTorch implementation of **Domain-Adversarial and Decision Distribution Alignment (DADDA)** for unsupervised domain adaptation. This repository demonstrates the core components and training procedure of the DADDA algorithm proposed in our paper.

> **Note**
> This repository is intended as an educational implementation of DADDA. It focuses on the algorithm itself rather than providing a complete research framework, benchmark suite, or production-ready training pipeline.

---

## Overview

Deep learning models for additive manufacturing often suffer from **domain shift**, where a model trained on one manufacturing condition performs poorly on another due to differences in machines, materials, process parameters, or sensing conditions.

DADDA addresses this problem by combining two complementary objectives:

- **Domain-adversarial learning**, which encourages the feature extractor to learn domain-invariant representations.
- **Decision distribution alignment**, which aligns class-conditional decision boundaries by minimizing the discrepancy between two task classifiers.

The combination improves transferability while preserving discriminative representations for defect classification.

---

## DADDA Architecture

```
                 Source Images                 Target Images
                       │                             │
                       └──────────────┬──────────────┘
                                      │
                              Feature Extractor
                                      │
                      ┌───────────────┴───────────────┐
                      │                               │
               Task Classifier 1              Task Classifier 2
                      │                               │
                      └──── Symmetric KL Divergence ──┘
                                      │
                              Domain Classifier
```

---

## Repository Structure

```text
dadda-domain-adaptation/
│
├── notebooks/
│   └── dadda_experiments.ipynb
│
├── src/
│   ├── data/
│   │   ├── preprocessing.py
│   │   ├── dataset.py
│   │   └── loaders.py
│   │
│   ├── models/
│   │   ├── encoder.py
│   │   ├── classifiers.py
│   │   ├── domain_classifier.py
│   │   └── dadda.py
│   │
│   ├── losses/
│   │   ├── discrepancy.py
│   │   └── domain_adversarial.py
│   │
│   └── training/
│       └── train_dadda.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Training Pipeline

The notebook `notebooks/dadda_experiments.ipynb` demonstrates the complete workflow:

1. Load and preprocess source and target datasets.
2. Construct PyTorch datasets and dataloaders.
3. Build the DADDA model.
4. Train the model using:
   - source classification loss,
   - adversarial domain loss,
   - symmetric KL discrepancy loss.
5. Visualize the training losses.

The notebook presents a simplified implementation of DADDA intended to demonstrate the key concepts of the algorithm. Several engineering details, optimizations, and experimental settings from the original research code have been omitted for clarity.

---

## Main Components

### Feature Extractor

A convolutional neural network that learns a shared latent representation for both source and target domains.

### Dual Task Classifiers

Two independent classifiers operate on the shared feature representation. Their prediction discrepancy is measured using symmetric KL divergence to encourage class-conditional alignment.

### Domain Classifier

A binary classifier trained adversarially to distinguish source and target domains, encouraging domain-invariant feature learning.

---

## Loss Functions

The implementation includes two core losses:

- **Symmetric KL Discrepancy Loss**
  - Measures disagreement between the two task classifiers on target samples.
  - Encourages class-conditional decision distribution alignment.

- **Domain Adversarial Loss**
  - Encourages the learned features to be indistinguishable across domains.

---

## Requirements

- Python 3.10+
- PyTorch
- NumPy
- OpenCV
- scikit-learn
- Matplotlib
- tqdm

Install all dependencies using

```bash
pip install -r requirements.txt
```

---

## Citation

If you find this implementation useful, please cite:

```bibtex
@article{xie2026reusability,
  title={On the reusability of machine learning-based process monitoring systems for manufacturing digital twins},
  author={Xie, Jiarui and Yang, Zhuo and Yang, Haw-Ching and Lu, Yan and Zhao, Yaoyao Fiona},
  journal={Journal of Manufacturing Systems},
  volume={84},
  pages={333--356},
  year={2026},
  publisher={Elsevier}
}
```