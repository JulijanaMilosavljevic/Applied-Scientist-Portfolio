# 🌸 04 — Transfer Learning with MobileNetV2 (Flower Classification)

Fine-tuned MobileNetV2 model trained on the Flowers Recognition dataset  
(5 classes: daisy, rose, sunflower, tulip, dandelion).  
This project demonstrates a modern Transfer Learning pipeline used in Applied Scientist workflows:

- 🌱 Data preprocessing & augmentation  
- ⚙️ Transfer learning with ImageNet weights  
- 🔧 Fine-tuning top MobileNetV2 layers  
- 📊 Evaluation: accuracy, confusion matrix, classification report  
- 🔥 Grad-CAM interpretability  
- 🧱 Clean project structure for industry interviews  
## 📁 Project Structure

04-cv-transfer-learning/  
├── data/  
│   └── flowers/  
├── models/  
│   └── final_mobilenetv2_flowers.keras  
├── plots/  
│   ├── training_curves.png  
│   └── confusion_matrix.png  
├── gradcam/  
│   ├── example.png  
│   └── gradcam_grid.png  
├── 04-cv-transfer-learning.ipynb  
├── README.md  
└── requirements.txt
## 🧠 1. Problem Description

The dataset contains **5 flower species**, each with unique texture, shape, and color.  
The goal is to build an accurate classifier capable of identifying the flower type from an input image.

### 🔍 Challenges:
- High color variability  
- Similar shapes across classes  
- Non-uniform backgrounds  
- Class imbalance  

This makes the dataset ideal for **Transfer Learning**, especially with lightweight architectures like **MobileNetV2**.
## 🌱 2. Data Loading & Augmentation

Images are preprocessed using `ImageDataGenerator` with:

- random rotations  
- width/height shifts  
- zoom  
- horizontal flips  
- brightness variation  
- rescaling (1/255)  

Dataset split:
- **80% training**
- **20% validation**

All images resized to **160×160**.

Augmentation significantly improves generalization and reduces overfitting.
## 🧊 3. Transfer Learning Setup (MobileNetV2)

We load **MobileNetV2 pretrained on ImageNet** with:

```python
include_top=False
weights="imagenet"
input_shape=(160, 160, 3)
```
Then we stack:

- GlobalAveragePooling2D

- Dropout(0.3)

- Dense(num_classes, activation="softmax")

✔ Why MobileNetV2?

- lightweight

- high accuracy

- efficient for fine-tuning

- ideal for medium-sized image datasets

---
## 🏋️‍♀️ 4. Training — Stage 1 (Frozen Base)

We first freeze the entire MobileNetV2 base and train only the classifier head:

- optimizer: `Adam(1e-3)`
- epochs: ~10
- purpose: stabilize top layers before unfreezing

This stage quickly increases accuracy without affecting pretrained weights.
## 🔥 5. Fine-Tuning — Stage 2 (Unfrozen)

After initial convergence, we unfreeze the **top 30 MobileNetV2 layers** and continue training:

- lower learning rate (`1e-4`)
- early stopping enabled
- validation monitored
- best model checkpoint saved

Fine-tuning significantly improves classification accuracy.
## 📈 6. Training Curves

Training and validation curves show stable improvement during fine-tuning.

📌 `plots/training_curves.png`
## 🧪 7. Evaluation Metrics

Evaluation includes:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- Confusion matrix

📌 `plots/confusion_matrix.png`

The model performs strongly on visually distinct classes such as **sunflower** and **rose**, with balanced results across all labels.
## 🌡️ 8. Grad-CAM Interpretability

Grad-CAM heatmaps reveal which image regions contribute to predictions.

Findings:
- model focuses on petals, centers, and color patterns  
- background has minimal influence  
- different classes show unique activation signatures  

📌 Example heatmaps:
- `gradcam/example.png`
- `gradcam/gradcam_grid.png`
## 💾 9. Saving the Model

The final fine-tuned model is saved as:

models/final_mobilenetv2_flowers.keras

Suitable for deployment or loading for inference.
## 📝 10. How to Run

```bash
git clone https://github.com/JulijanaMilosavljevic/Applied-Scientist-Portfolio
cd 04-cv-transfer-learning
pip install -r requirements.txt
jupyter notebook 04-cv-transfer-learning.ipynb
```
## 🎯 Final Notes

This project demonstrates essential Applied Scientist skills:

- CNNs & Transfer Learning  
- Fine-tuning deep architectures  
- Data augmentation  
- Model evaluation & interpretability  
- Clean ML pipeline  
