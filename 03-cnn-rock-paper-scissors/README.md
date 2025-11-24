# 📘 Rock–Paper–Scissors Image Classification (CNN, TensorFlow)

This project implements a **Convolutional Neural Network (CNN)** that classifies hand gesture images into **Rock, Paper, or Scissors**.  
It demonstrates a complete deep-learning workflow including:

- image preprocessing  
- data augmentation  
- CNN model development  
- training and validation  
- evaluation on a held-out test set  
- Grad-CAM model interpretability  
- sample predictions  

This project is part of my **Applied Scientist Portfolio**.

---

## 📂 Project Structure

```bash
03-cnn-rps/
│
├── data/                # Train / valid / test folders
├── models/              # Saved model (.keras)
├── 03-cnn-rock-paper-scissors.ipynb     # Full notebook
└── README.md            # (this file)
```

## 🧠 1. Problem Description

The goal of this project is to classify hand gesture images into:

- ✊ **Rock**
- ✋ **Paper**
- ✌️ **Scissors**

The dataset contains images with:

- diverse skin tones  
- different lighting environments  
- various hand positions  
- multiple backgrounds  

This makes the classification task realistic and suitable for testing generalization.

---

## 🧹 2. Dataset & Preprocessing

**Dataset:** Rock–Paper–Scissors (Roboflow version)

- **Train:** 70%  
- **Validation:** 20%  
- **Test:** 10%  
- **Image size:** 100×100  
- **Scaling:** pixel values normalized to `[0, 1]`

### 🌀 Data Augmentation

```python
rotation_range=20
width_shift_range=0.1
height_shift_range=0.1
zoom_range=0.1
horizontal_flip=True
fill_mode="nearest"
```
This increases robustness and prevents overfitting.

---

## 🧱 3. CNN Model Architecture

The model is built using the **TensorFlow Functional API**, with the following structure:

Input (100x100x3)
│
├── Conv2D(32) + BatchNorm + MaxPool
├── Conv2D(64) + BatchNorm + MaxPool
├── Conv2D(128) + BatchNorm + MaxPool   ← last conv layer (for Grad-CAM)
│
├── Flatten
├── Dense(128, relu) + Dropout(0.4)
└── Dense(3, softmax)



### 🔧 Training Setup

- **Loss:** `categorical_crossentropy`  
- **Optimizer:** `Adam`  
- **Metrics:** `accuracy`  
- **Callbacks:**  
  - `EarlyStopping`  
  - `ModelCheckpoint`  

---

## 📈 4. Training Curves

Below are the training and validation accuracy/loss plots:

*(Insert accuracy/loss plots here)*

### ✔ Observations

- Smooth and stable convergence  
- No overfitting (both curves follow each other closely)  
- Validation accuracy stabilizes around **97–99%**  
- Excellent generalization to unseen data  

---

## 🧪 5. Evaluation on Test Set

### ✔ Confusion Matrix

*(Insert confusion matrix image here)*

All classes were predicted **perfectly**:

- **Paper:** 30 / 30  
- **Rock:** 28 / 28  
- **Scissors:** 33 / 33  

### ✔ Classification Report

| Class     | Precision | Recall | F1-score |
|-----------|-----------|--------|----------|
| paper     | 1.00      | 1.00   | 1.00     |
| rock      | 1.00      | 1.00   | 1.00     |
| scissors  | 1.00      | 1.00   | 1.00     |

➡️ **Overall Accuracy: 100%**

This level of performance is expected because the classes are visually distinct and the CNN is well-trained.

---

## 🔍 6. Grad-CAM Visualization

Grad-CAM highlights which areas of the image the model considers most important for classification.

*(Insert Original → Heatmap → Overlay images)*

### ✔ Insights

The heatmaps show that the model focuses on:

- finger contours  
- gesture shape  
- hand silhouette  

It ignores the background — confirming that the CNN learned **true gesture-based features**, not color or noise.

---

## 🖼️ 7. Sample Predictions

Below are random predictions generated on unseen test images:

*(Insert 6-image prediction grid)*

### ✔ Generalization Behavior

The model performs correctly across:

- different skin tones  
- different lighting  
- different backgrounds  
- different hand angles  

This demonstrates strong robustness and discriminative power.

---

## 📝 8. Final Conclusions

- The CNN achieves **perfect test-set accuracy** on this dataset.  
- Training is stable with no signs of overfitting.  
- Model interpretability via Grad-CAM confirms correct behavior.  
- This project demonstrates key **Applied Scientist skills**, including:
  - computer vision preprocessing  
  - data augmentation  
  - CNN architecture design  
  - evaluation techniques  
  - interpretability for deep learning models  

---

## 🛠️ 9. Reproduce This Project

Clone the portfolio:

```bash
git clone https://github.com/JulijanaMilosavljevic/Applied-Scientist-Portfolio
cd Applied-Scientist-Portfolio/03-cnn-rock-paper-scissors
```

Run notebook:
```
pip install -r requirements.txt
jupyter notebook 03-cnn-rock-paper-scissors.ipynb
```
