# Predict Fuel Efficiency Using TensorFlow in Python

A deep learning regression project that predicts a car's **fuel efficiency (Miles Per Gallon - MPG)** using TensorFlow and Keras. The model is trained on the Auto MPG dataset by learning patterns from vehicle specifications such as horsepower, weight, cylinders, and acceleration.

---

## Project Overview

This notebook covers the complete machine learning pipeline:

- Data loading and cleaning
- Exploratory Data Analysis (EDA)
- Feature selection
- Training & validation split
- TensorFlow data pipeline
- Deep neural network model
- Model evaluation using MAE & MAPE
- Training loss visualization

---

## Dataset

**Dataset:** Auto MPG Dataset

The dataset contains **398 vehicle records** with 9 attributes. After removing rows with missing horsepower values, **392 samples** were used for training.

### Features Used

| Feature | Description |
|----------|-------------|
| Cylinders | Number of engine cylinders |
| Horsepower | Engine power |
| Weight | Vehicle weight |
| Acceleration | Acceleration time |
| Model Year | Manufacturing year |
| Origin | Country of origin |

**Target Variable:** `mpg` (Miles Per Gallon)

> `car name` was removed and `displacement` was dropped due to high correlation with other features.

---

## Tech Stack

- Python
- TensorFlow & Keras
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Model Architecture

The neural network consists of:

- Dense (256, ReLU)
- Batch Normalization
- Dense (256, ReLU)
- Dropout (0.3)
- Batch Normalization
- Dense (1)

**Loss Function:** Mean Absolute Error (MAE)

**Optimizer:** Adam

**Metric:** Mean Absolute Percentage Error (MAPE)

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Dataset Size | 392 samples |
| Training Split | 80% |
| Validation Split | 20% |
| Epochs | 50 |
| Batch Size | 32 |
| Optimizer | Adam |
| Loss | MAE |

The model was trained for **50 epochs**, and both training and validation metrics were monitored throughout the process.

---

## Project Structure

```text
Predict_Fuel_Efficiency_Using_Tensorflow_in_Python.ipynb
README.md
auto-mpg.csv
```

---

## How to Run

1. Clone this repository.
2. Install the required libraries:

```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
```

3. Open the notebook:

```bash
jupyter notebook Predict_Fuel_Efficiency_Using_Tensorflow_in_Python.ipynb
```

4. Run all cells sequentially.

---

## Learning Outcomes

- Data preprocessing for regression problems
- Handling missing values
- Correlation analysis & feature selection
- Building deep neural networks with TensorFlow
- Using Batch Normalization and Dropout
- Evaluating regression models with MAE and MAPE

---

## Author

**Vidushi Srivastava**
