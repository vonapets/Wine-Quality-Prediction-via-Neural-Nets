# Wine Quality Neural Network Experiments

This project compares the performance of neural networks built in **TensorFlow/Keras** and **PyTorch** for predicting wine quality based on physicochemical measurements.

The dataset used contains chemical measurements of wines (e.g., acidity, sugar, pH, sulfates, alcohol) and a quality rating from 0–10. The task is formulated as a supervised classification problem.

---

## 📌 Project Goals

* Build and compare feed-forward neural networks in TensorFlow and PyTorch
* Conduct multiple controlled experiments to measure the effect of:

  * Learning rate
  * Activation function
  * Number of layers
  * Batch size
  * Early stopping
* Visualize training metrics over time
* Generate 3D PCA plots to understand data separability

---

## 🧠 Methods & Experiments

Multiple experiments were automated, including:

| Experiment | Variable Tested                    |
| ---------- | ---------------------------------- |
| Control    | Baseline model                     |
| Q1-a       | Lower learning rate (0.0001)       |
| Q1-b       | Higher learning rate (0.01)        |
| Q2         | Activation function (tanh vs relu) |
| Q3         | Depth (2 vs 4 layers)              |
| Q4-a       | Larger batch size (128)            |
| Q4-b       | Smaller batch size (8)             |
| Q5         | Early stopping                     |

Each run logged:

* Train loss
* Validation loss
* Accuracy (train/val/test)
* AUC score

All results were plotted and printed in tables for comparison.

---

## 🔍 Key Findings (Summary)

* **Best accuracy:**
  PyTorch, 4-layer model (≈ **91.25%**)
* **Best AUC calibration:**
  TensorFlow, batch size 128 (≈ **0.917**)
* Higher layer count helps PyTorch more than TensorFlow
* Larger batch sizes produced more stable learning and better probability calibration
* Early stopping cut training time by ~80% with minimal performance loss

---

## 📊 Visualizations

The script generates:

* Training vs validation loss plots
* Accuracy curves across training epochs
* AUC improvement graphs
* 3D PCA scatter plots of the dataset

These help explain why the problem is not perfectly separable—many wines with similar chemical profiles have different final quality scores.

---

## 🗂 Project Structure

```
├── assignment_3_vlad_stepanov_gb657.py   # Main script
├── results/                              # (Optional) Images & exported tables
└── data/                                 # Dataset (optional if included)
```

---

## ▶️ How to Run

1. Install required libraries:

```
pip install numpy pandas matplotlib scikit-learn tensorflow torch statsmodels
```

2. Ensure the dataset CSV is placed in the expected directory or adjust its path inside the script.

3. Run the script:

```
python assignment_3_vlad_stepanov_gb657.py
```

---

## 📁 Dataset

The script currently loads:

```
Assignment_1_winequality (2).csv
```

---

## 🏆 Author

**Vlad Stepanov**
Graduate assignment for GB657 – Machine Learning
