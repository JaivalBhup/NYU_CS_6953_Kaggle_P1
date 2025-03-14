## CIFAR-10 Image Classification with ResNet and SE Blocks
This repository is the the submission for the NYU CS 6953 Project 1

## Introduction

This project implements a **ResNet-based model** enhanced with **Squeeze-and-Excitation (SE) Blocks** for image classification on the **CIFAR-10 dataset**. The model is trained using **SGD with Nesterov momentum**, **Cosine Annealing learning rate scheduling**, and **Kaiming weight initialization** to improve convergence and accuracy.

The training pipeline automatically downloads CIFAR-10, preprocesses the data, trains the model, and saves the final model in the `./model_save` directory.

---
## Repository
- models.py: ResNet model architechture
- main.py: File to train and test the mode
- Python Notebooks: Folder containing .ipynb files

--- 
## Prerequisites

Ensure you have **Python 3.6+** installed before running the project.

To install the required dependencies, run:

```sh
pip install -r requirements.txt
```

---

## Training the Model

Run the following command to start training:

```sh
python main.py
```

This will:

- Download and preprocess the CIFAR-10 dataset.
- Train the model with the specified hyperparameters.
- Save the trained model in `./model_save/`.

---

## Hyperparameters

Below is the summary of the key hyperparameters used in training:

## 🔧 Hyperparameters

| **Hyperparameter**         | **Value**                      |
|---------------------------|--------------------------------|
| ResNet Layers             | 3                              |
| Block Size                | [4, 4, 3]                      |
| Squeeze and Excitation    | True                           |
| Channels (Ci)             | [64, 128, 256]                 |
| Filter Size (Fi)          | [3, 3, 3]                      |
| Skip Kernel Size (Ki)     | [1, 1, 1]                      |
| Average Pool Size (Pi)    | 8                              |
| Optimizer                 | SGD                            |
| Nesterov                  | True                           |
| Lookahead                 | False                          |
| Momentum                  | 0.9                            |
| Learning Rate (LR)        | 0.1                            |
| LR Scheduler              | CosineAnnealing, η_min=0.0001  |
| Weight Decay              | 0.0005                         |
| Epochs                    | 100                            |
| Batch Size                | 128                            |
| Data Augmentation         | True                           |

---

## Model Checkpoints

Trained models will be saved in:

```
./model_save/
```

You can load the model later for inference or fine-tuning.

---

## Results

The model achieves a **validation accuracy of 95.2%**, demonstrating the effectiveness of SE Blocks and optimized training strategies.

---

## References

- Thakur, Aditya, et al. 2023 “Efficient ResNets: Residual Network Design.” https://arxiv.org/abs/2306.12100
- Hu, Jie, et al. 2017 “Squeeze-and-Excitation Networks.” https://arxiv.org/abs/1709.01507
- Perez, Luis, and Jason Wang. 2017 “The Effectiveness of Data Augmentation in Image Classification using Deep Learning.” https://arxiv.org/abs/1712.04621
- He, Kaiming, et al. 2016 “Identity Mappings in Deep Residual Networks.” https://arxiv.org/pdf/1603.05027.
- Zhang, Michael, et al. 2020. “Lookahead Optimizer: Escape Local Minima and Improve Generalization.” ICML 2020 https://arxiv.org/abs/2006.12007
- Pereyra, G., et al. 2017. “Regularizing Neural Networks by Penalizing Confident Outputs.” ICLR 2017. https://arxiv.org/abs/1701.06548
- Kidambi, R., et al. 2018. “On the Convergence of Nesterov’s Accelerated Gradient Method in Stochastic Settings.”ICML 2018.https://arxiv.org/abs/1805.0906
