# 🍎 Fruit Detection and Classification

A deep learning project for detecting and classifying fruits using **Transfer Learning with MobileNet** on TensorFlow/Keras. The model is trained to identify 9 different types of fruits from images with high accuracy.

---

## 📁 Dataset

**Source:** [Fruits Dataset Images — Kaggle](https://www.kaggle.com/datasets/shreyapmaher/fruits-dataset-images)

The dataset contains labeled images of 9 fruit categories organized in class-wise folders.

**Fruit Classes:**
- Apple
- Banana
- Cherry
- Chickoo
- Grapes
- Kiwi
- Mango
- Orange
- Strawberry

---

## 🧠 Model Architecture

- **Base Model:** MobileNet (pretrained on ImageNet, top layers excluded)
- **Custom Head:**
  - GlobalAveragePooling2D
  - Dense (1024 units, ReLU activation)
  - Dense (9 units, Softmax activation)
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Input Shape:** 224 × 224 × 3

The base MobileNet layers are **frozen** during training (feature extraction approach).

---

## 🔄 Workflow

```
Dataset Download (KaggleHub)
        ↓
Train/Test Split (80/20)
        ↓
Data Augmentation (rotation, zoom, flip, shift)
        ↓
Model Building (MobileNet + Custom Head)
        ↓
Model Training (10 epochs)
        ↓
Evaluation (Accuracy & Loss plots)
        ↓
Prediction on New Images
```

---

## 📊 Data Augmentation

Applied only on the **training set**:
- Rotation (±20°)
- Width & Height Shift (20%)
- Shear & Zoom (20%)
- Horizontal Flip
- Fill Mode: Nearest

Test/Validation set uses only **rescaling (1/255)**.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/fruit-detection-classification.git
cd fruit-detection-classification
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

```python
import kagglehub
path = kagglehub.dataset_download("shreyapmaher/fruits-dataset-images")
```

> Make sure your Kaggle API key is configured before running this.

### 4. Run the Notebook

Open `Fruits_Detection_and_Classification.ipynb` in **Google Colab** or **Jupyter Notebook** and run all cells sequentially.

---

## 📂 Project Structure

```
fruit-detection-classification/
│
├── Fruits_Detection_and_Classification.ipynb   # Main notebook
├── fruit_classifier_mobilenet.h5               # Saved trained model
├── requirements.txt                            # Python dependencies
└── README.md                                   # Project documentation
```

---

## 🖼️ Prediction Example

```python
show_prediction('/content/test/mango fruit/Image_14.jpg', model, class_names)
```

This displays the image with the predicted fruit label and confidence score.

---

## 📈 Results

After training for **10 epochs**, the model generates:
- **Training & Validation Accuracy** curve
- **Training & Validation Loss** curve

Plots are visualized using Matplotlib.

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| Python | Core programming language |
| TensorFlow / Keras | Deep learning framework |
| MobileNet | Pretrained base model |
| KaggleHub | Dataset download |
| Scikit-learn | Train/test split |
| Matplotlib | Visualization |
| NumPy | Numerical operations |

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
