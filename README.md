# 🔢 Handwritten Digit Recognition using Support Vector Machine (SVM)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange?logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Accuracy](https://img.shields.io/badge/Accuracy-~98%25-brightgreen)

---

## 📌 Project Overview

This project builds a **handwritten digit recognition classifier** using a **Support Vector Machine (SVM)** trained on the classic **Scikit-learn Digits dataset**. The model accurately classifies digits 0–9 from 8×8 pixel grayscale images and is persisted using Python's `pickle` module for reuse without retraining.

This is an ideal beginner-to-intermediate machine learning project that demonstrates the full ML pipeline: data exploration → preprocessing → model training → evaluation → model persistence.

---

## 📁 Project Structure

```
digit-svm-classifier/
│
├── digit_SVM_Classifier.ipynb   # Main Jupyter notebook
├── model_pickle                 # Saved trained SVM model
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
│
└── visualizations/              # Generated plots (after running notebook)
    ├── sample_digits.png
    ├── class_distribution.png
    ├── avg_pixel_heatmap.png
    ├── confusion_matrix.png
    ├── classification_report_heatmap.png
    ├── misclassified_samples.png
    ├── pca_scatter.png
    └── kernel_comparison.png
```

---

## 📊 Dataset

| Property         | Details                           |
|------------------|-----------------------------------|
| **Source**       | `sklearn.datasets.load_digits()`  |
| **Samples**      | 1,797                             |
| **Features**     | 64 (8×8 pixel flattened image)    |
| **Classes**      | 10 (digits 0–9)                   |
| **Class Balance**| ~180 samples per class (balanced) |

Each image is an 8×8 grayscale grid stored as a 64-dimensional feature vector with pixel values ranging from 0 to 16.

---

## ⚙️ Tech Stack

| Tool/Library     | Purpose                              |
|------------------|--------------------------------------|
| Python 3.8+      | Core programming language            |
| NumPy            | Numerical operations                 |
| Pandas           | Data manipulation & DataFrame ops    |
| Scikit-learn     | SVM model, train-test split, metrics |
| Matplotlib       | Visualizations                       |
| Seaborn          | Enhanced statistical plots           |
| Pickle           | Model serialization                  |

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/digit-svm-classifier.git
cd digit-svm-classifier
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Notebook
```bash
jupyter notebook digit_SVM_Classifier.ipynb
```

---

## 📋 requirements.txt

```
numpy
pandas
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 🔬 Methodology

### Step 1 — Data Loading & Exploration
- Loaded the Digits dataset via `sklearn.datasets.load_digits()`
- Created a Pandas DataFrame with 64 pixel features + target column
- Verified no missing values exist in the dataset

### Step 2 — Feature & Target Split
- `X` → 64 pixel intensity features
- `y` → Digit labels (0–9)

### Step 3 — Train-Test Split
- 70% training / 30% testing split
- `random_state=42` ensures reproducibility

### Step 4 — Model Training
- Used `sklearn.svm.SVC()` with default RBF kernel
- Fit on training data

### Step 5 — Evaluation
- Achieved **~98% accuracy** on the test set
- Visualized performance via Confusion Matrix and Classification Report

### Step 6 — Model Persistence
- Model saved with `pickle` as `model_pickle`
- Can be reloaded and reused without retraining

---

## 📈 Results

| Metric       | Score   |
|--------------|---------|
| **Accuracy** | ~98%    |
| **Precision**| ~98%    |
| **Recall**   | ~98%    |
| **F1-Score** | ~98%    |

The SVM with RBF kernel achieved excellent performance on the Digits dataset, correctly classifying nearly all test samples.

---

## 📷 Visualizations

The following visualizations are included in the notebook:

| Visualization                    | Description                                              |
|----------------------------------|----------------------------------------------------------|
| Sample Digit Images              | Grid of 20 sample images from the dataset                |
| Class Distribution Bar Chart     | Count of samples per digit class                         |
| Average Pixel Intensity Heatmap  | Mean image representation for each digit (0–9)           |
| Confusion Matrix                 | Predicted vs. actual classification results              |
| Classification Report Heatmap    | Precision, Recall, F1-Score per class                    |
| Misclassified Samples            | Gallery of incorrectly predicted digit images            |
| PCA 2D Scatter Plot              | Dimensionality-reduced visualization of digit clusters   |
| SVM Kernel Comparison            | Accuracy comparison across Linear, RBF, Poly, Sigmoid    |

---

## 💾 Model Persistence

The trained model is saved using `pickle`:

```python
import pickle

# Save model
with open("model_pickle", "wb") as f:
    pickle.dump(model, f)

# Load model
with open("model_pickle", "rb") as f:
    model = pickle.load(f)

# Predict
prediction = model.predict([sample_image_features])
```

---

## 🧠 Key Learnings

- SVM is highly effective on small, well-structured datasets like handwritten digits
- The RBF (Radial Basis Function) kernel handles non-linear decision boundaries in pixel feature spaces
- PCA reveals that digit clusters are well-separated in lower dimensions
- Model persistence with `pickle` enables production-ready deployment without retraining

---

## 🔮 Future Improvements

- [ ] Hyperparameter tuning with `GridSearchCV` (C, gamma, kernel)
- [ ] Compare SVM against other classifiers (Random Forest, KNN, CNN)
- [ ] Build a simple drawing interface (Tkinter or Streamlit) for live digit prediction
- [ ] Deploy the model as a REST API using Flask or FastAPI
- [ ] Extend to the MNIST dataset (70,000 full 28×28 images)

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**
- GitHub: https://github.com/MuhammadAsadKhan-11
- LinkedIn: www.linkedin.com/in/muhammad-asad-khan-9a70243a3

---

> ⭐ If you found this project helpful, please consider giving it a star!
