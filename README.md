# MLOps Unit 1 - Python ML Workflow

## 📌 Overview

This project demonstrates a basic Machine Learning workflow along with a standard MLOps-friendly project structure.

It includes:

* Data loading using pandas
* Train-test split
* Model training using Logistic Regression
* Model evaluation
* Model saving using joblib

---

## 📁 Project Structure

```
mlops-unit1/
│
├── data/              # Dataset storage (currently using sklearn dataset)
├── src/               # Source code
│   └── train_model.py
├── models/            # Saved ML models
│   └── model.joblib
├── requirements.txt   # Project dependencies
└── README.md
```

---

## ⚙️ Installation

Install dependencies using:

```
pip install -r requirements.txt
```

---

## ▶️ How to Run

Run the training script:

```
python src/train_model.py
```

---

## 📊 Model Details

* Model Used: Logistic Regression
* Dataset: Iris Dataset (from sklearn)
* Evaluation Metrics:

  * Accuracy Score
  * Classification Report

---

## 💾 Model Saving

The trained model is saved in:

```
models/model.joblib
```

This allows reuse without retraining.

---

## 🚀 Key Concepts Demonstrated

* Machine Learning workflow
* Data preprocessing
* Model training & evaluation
* Version control using Git
* MLOps project structuring

---

## 👨‍💻 Author

Himanshu Dhaka
BTech CSE (Core) | Aspiring Data Scientist
