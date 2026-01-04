# Fashion-MNIST Image Classification

A deep learning project using **TensorFlow & Keras** to classify Fashion-MNIST images into 10 clothing categories.

---

## 📌 Overview
This project implements the full deep learning workflow:
- Loading and exploring the Fashion-MNIST dataset
- Normalizing image data
- Building and training a neural network
- Evaluating model performance
- Visualizing predictions and model confidence

---

## 🧠 Model
- Flatten (28×28 → 784)
- Dense (128 neurons, ReLU)
- Dense (10 neurons, Softmax)

---

## 📊 Results
- **Training Accuracy:** ~91%
- **Test Accuracy:** ~87–88%
- Mild and expected overfitting

---

## 🖼️ Visualizations
The project includes visual outputs showing:
- Sample images from the dataset
- Predicted vs true labels
- Probability distribution for each prediction

---

## ⚙️ Tools
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

---

## 🚀 Note
This project represents my **first hands-on deep learning implementation**, focused on understanding core concepts and the complete pipeline.

## 🖼️ Visual Results

### Dataset Samples
The following image shows sample images from the Fashion-MNIST dataset with their corresponding class labels.

![Dataset Samples](img/pic1.png)

---

### Model Predictions
The image below shows the model’s predictions, including the predicted class and confidence distribution for each sample.

![Prediction Results](img/pic2.png)
