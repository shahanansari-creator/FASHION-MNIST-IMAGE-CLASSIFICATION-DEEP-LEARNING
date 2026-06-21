# FASHION-MNIST-IMAGE-CLASSIFICATION-DEEP-LEARNING

# 👗 Fashion MNIST — Image Classification with FCNN

> A Fully Connected Neural Network (FCNN) built with TensorFlow/Keras to classify grayscale fashion images into 10 categories — trained and evaluated on Google Colab.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Model Architecture](#-model-architecture)
- [Preprocessing Pipeline](#-preprocessing-pipeline)
- [Training Configuration](#-training-configuration)
- [Results](#-results)
- [Visualizations](#-visualizations)
- [Getting Started](#-getting-started)
- [Requirements](#-requirements)
- [Key Design Decisions](#-key-design-decisions)
- [Future Improvements](#-future-improvements)

---

## 🧠 Overview

This project implements a **Fully Connected Neural Network (FCNN)** — also known as a Multi-Layer Perceptron (MLP) — to classify images from the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset. The goal is to correctly identify each 28×28 grayscale image as one of **10 fashion item categories**.

The entire workflow covers:
- 📊 Data visualization and distribution analysis
- 🔧 Preprocessing (normalization, flattening, one-hot encoding)
- 🏗️ FCNN model design with regularization (Dropout + Batch Normalization)
- 🏋️ Training with SGD optimizer and categorical cross-entropy loss
- 📈 Performance evaluation via accuracy curves, confusion matrix, and classification report

---

## 📦 Dataset

**Fashion MNIST** was introduced by Zalando Research as a harder drop-in replacement for the classic handwritten digit MNIST dataset.

| Property | Details |
|---|---|
| Total Images | 70,000 (60,000 train + 10,000 test) |
| Image Size | 28 × 28 pixels, grayscale |
| Pixel Range | 0 – 255 (uint8) |
| Number of Classes | 10 |
| Class Balance | Perfectly balanced (6,000 samples/class in training) |
| Source | `tf.keras.datasets.fashion_mnist` — built-in, no download needed |

### 🏷️ Class Labels

| Index | Category |
|:---:|---|
| 0 | T-shirt / Top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle Boot |

---

## 📁 Project Structure

```
fashion-mnist-fcnn/
│
├── fashion_mnist_fcnn.py       # Main script (all steps end-to-end)
│
├── outputs/                    # Auto-generated when you run the script
│   ├── sample_images.png       # One image per class (2×5 grid)
│   ├── class_distribution.png  # Bar chart of training class counts
│   ├── training_history.png    # Loss & accuracy curves (train vs val)
│   ├── confusion_matrix.png    # 10×10 heatmap on test set
│   └── predictions.png         # 15 random test predictions
│
└── README.md
```

---

## 🏗️ Model Architecture

```
Input (784)
    │
    ▼
Dense(256, ReLU)
    │
BatchNormalization
    │
Dropout(0.3)
    │
    ▼
Dense(128, ReLU)
    │
BatchNormalization
    │
Dropout(0.3)
    │
    ▼
Dense(64, ReLU)
    │
    ▼
Dense(10, Softmax)  ← Output: probability over 10 classes
```

### Layer Details

| Layer | Config | Purpose |
|---|---|---|
| Input | 784 neurons | One neuron per flattened pixel |
| Dense 1 | 256 units, ReLU | Low-level feature extraction |
| BatchNorm 1 | — | Stabilizes activations, speeds training |
| Dropout 1 | rate = 0.3 | Drops 30% of neurons → prevents overfitting |
| Dense 2 | 128 units, ReLU | Mid-level feature extraction |
| BatchNorm 2 | — | Applied after second dense layer |
| Dropout 2 | rate = 0.3 | Second regularization layer |
| Dense 3 | 64 units, ReLU | High-level feature refinement |
| Output | 10 units, Softmax | Class probability distribution |

> **Total trainable parameters:** ~240,000+

---

## 🔧 Preprocessing Pipeline

### 1. Normalization
Pixel values scaled from `[0, 255]` → `[0.0, 1.0]`
```python
X_train = X_train / 255.0
X_test  = X_test  / 255.0
```

### 2. Flattening
2D images reshaped to 1D vectors for the FCNN input layer
```python
X_train_flat = X_train.reshape(-1, 784)   # (60000, 28, 28) → (60000, 784)
X_test_flat  = X_test.reshape(-1, 784)    # (10000, 28, 28) → (10000, 784)
```

### 3. One-Hot Encoding
Integer labels converted to binary vectors
```python
y_train_ohe = to_categorical(y_train, num_classes=10)
# e.g. class 3 → [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

---

## 🏋️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | SGD (Stochastic Gradient Descent) |
| Loss Function | Categorical Cross-Entropy |
| Metric | Accuracy |
| Epochs | 30 |
| Batch Size | 64 |
| Validation Split | 10% (6,000 images held out per epoch) |
| Effective Training Samples | 54,000 |

---

## 📊 Results

| Metric | Value |
|---|---|
| Expected Test Accuracy | **~88 – 90%** |
| Strongest Classes | Trouser, Bag, Ankle Boot |
| Hardest-to-Classify | Shirt vs T-shirt/Top, Coat vs Pullover |

Performance metrics are reported per class using `sklearn.metrics.classification_report`:
- **Precision** — correctness of positive predictions
- **Recall** — coverage of actual positives
- **F1-Score** — harmonic mean of precision and recall

---

## 🖼️ Visualizations

The script automatically generates and saves the following plots:

| File | Description |
|---|---|
| `sample_images.png` | 2×5 grid — one image per fashion class |
| `class_distribution.png` | Bar chart confirming balanced class counts |
| `training_history.png` | Train vs validation loss & accuracy over 30 epochs |
| `confusion_matrix.png` | 10×10 Seaborn heatmap on 10,000 test images |
| `predictions.png` | 15 random predictions — 🟢 green = correct, 🔴 red = wrong |

---

## 🚀 Getting Started

### Run on Google Colab (Recommended)

All required libraries are pre-installed on Colab — no setup needed.

1. Open [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Copy and paste the contents of `fashion_mnist_fcnn.py` into a code cell
4. Click **Runtime → Run All**

### Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/fashion-mnist-fcnn.git
cd fashion-mnist-fcnn

# Install dependencies
pip install tensorflow numpy matplotlib seaborn scikit-learn

# Run the script
python fashion_mnist_fcnn.py
```

---

## 📋 Requirements

```
tensorflow>=2.0
numpy
matplotlib
seaborn
scikit-learn
```

Or install all at once:
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

---

## 💡 Key Design Decisions

| Decision | Rationale |
|---|---|
| Layer sizes: 256 → 128 → 64 | Funnel architecture compresses features progressively |
| Dropout rate = 0.3 | Balances regularization without hurting learning |
| Batch Normalization after Dense | Normalizes before stochastic dropping; stabilizes training |
| SGD optimizer | Required by assignment; known to generalize well |
| 30 epochs | Enough for SGD to converge without major overfitting |
| Batch size = 64 | Balances speed and gradient noise |
| ReLU in hidden layers | Avoids vanishing gradients; standard for FCNNs |
| Softmax + CCE | Canonical pairing for multi-class classification |

---

## 🔮 Future Improvements

- [ ] Switch to **Adam optimizer** for faster convergence (~91–93% accuracy)
- [ ] Add **Learning Rate Scheduler** (e.g. `ReduceLROnPlateau`)
- [ ] Apply **Data Augmentation** (horizontal flips, small rotations)
- [ ] Upgrade to a **CNN** (Convolutional Neural Network) for 94%+ accuracy
- [ ] Add **EarlyStopping** callback to prevent unnecessary training epochs
- [ ] Experiment with deeper architectures and different dropout rates

---



<p align="center">
  Built with TensorFlow · Keras · Google Colab
</p>
