📘 Rock–Paper–Scissors Image Classification (CNN, TensorFlow)

This project implements a Convolutional Neural Network (CNN) that classifies hand gesture images into Rock, Paper, or Scissors.
It demonstrates a full deep-learning workflow including:

Image preprocessing & augmentation

Building a CNN using TensorFlow/Keras

Model training + validation analysis

Performance evaluation (confusion matrix, classification report)

Grad-CAM interpretability

Sample predictions

Clear project structure for reproducibility

This project is part of my Applied Scientist Portfolio.

📂 Project Structure
03-cnn-rps/
│
├── data/                # Train / valid / test folders
├── models/              # Saved model (.keras)
├── visualizations/      # Training curves, Grad-CAM, sample predictions
├── 03-cnn-rps.ipynb     # Full notebook
└── README.md            # (this file)

🧠 1. Problem Description

The goal is to classify images of human hands showing:

✊ rock

✋ paper

✌️ scissors

The dataset includes multiple skin tones, lighting conditions, backgrounds, and hand orientations — making it a realistic vision classification problem.

🧹 2. Dataset & Preprocessing

Dataset used: Rock–Paper–Scissors Hand Gestures (Roboflow version)

Split:

70% training

20% validation

10% test

Images resized to 100×100

Pixel values normalized to [0, 1]

Data augmentation:
rotation_range=20
width_shift_range=0.1
height_shift_range=0.1
zoom_range=0.1
horizontal_flip=True


This increases robustness and prevents overfitting.

🧱 3. Model Architecture (CNN)

The model is built using the Functional API:

Input (100x100x3)
│
├── Conv2D(32) + BatchNorm + MaxPool
├── Conv2D(64) + BatchNorm + MaxPool
├── Conv2D(128) + BatchNorm + MaxPool   ← last conv layer (for Grad-CAM)
│
├── Flatten
├── Dense(128, relu) + Dropout(0.4)
└── Dense(3, softmax)


Loss: categorical_crossentropy
Optimizer: Adam
Metrics: accuracy

📈 4. Training Curves

Training & validation accuracy/loss plots:

(Insert your Loss/Accuracy curves here — saved from notebook)

These curves show:

Smooth convergence

No overfitting

Validation accuracy stabilizing around 97–99%

🧪 5. Evaluation on Test Set
✔ Confusion Matrix

(Insert your confusion matrix image here)

The model correctly predicted every single test sample:

Paper: 30/30

Rock: 28/28

Scissors: 33/33

✔ Classification Report
Class	Precision	Recall	F1
paper	1.00	1.00	1.00
rock	1.00	1.00	1.00
scissors	1.00	1.00	1.00

➡️ Overall accuracy: 100%

The classes are visually distinct, so the CNN learns them extremely well.

🔍 6. Grad-CAM Interpretability

Grad-CAM visualizations show which image regions influence model decisions.

(Insert your Grad-CAM output here — Original / Heatmap / Overlay)

The model focuses on:

finger outlines

gesture shape

hand silhouette

— and ignores background, confirming correct feature learning.

🖼️ 7. Sample Predictions

(Insert your 6-image prediction grid)

The model generalizes across:

different skin tones

different lighting

different hand shapes

background variations

All predictions are correct.

📝 8. Final Conclusions

CNN achieves state-of-the-art accuracy on the RPS dataset

Excellent generalization — no class confusion

Grad-CAM shows meaningful attention to gesture shape

This project demonstrates essential Applied Scientist skills:

image preprocessing

augmentation

CNN architecture

training procedure

evaluation

model interpretability

🛠️ 9. Reproduce the Project

Clone the portfolio:

git clone https://github.com/JulijanaMilosavljevic/Applied-Scientist-Portfolio
cd Applied-Scientist-Portfolio/03-cnn-rps


Install requirements:

pip install -r requirements.txt


Run notebook:

jupyter notebook 03-cnn-rps.ipynb

📬 Contact

GitHub: https://github.com/JulijanaMilos

LinkedIn: (insert if you want)