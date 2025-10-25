# Iris Flower Classification - Machine Learning Project

## 📖 Project Overview
This project implements and compares multiple machine learning algorithms to classify iris flowers into three species based on their morphological characteristics.

## 📊 Dataset Information
The Iris dataset contains 150 samples of iris flowers with the following features:
- **Sepal Length** (cm)
- **Sepal Width** (cm) 
- **Petal Length** (cm)
- **Petal Width** (cm)

**Target Species:**
- 0: Setosa
- 1: Versicolor
- 2: Virginica

## 🛠️ Technologies Used
- **Python 3.x**
- **Scikit-learn** - Machine learning algorithms
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical data visualization

## 🤖 Machine Learning Algorithms Implemented

### 1. Logistic Regression
- A linear model for classification
- Uses probability to predict class membership
- Fast training and good interpretability

### 2. k-Nearest Neighbors (kNN)
- Instance-based learning algorithm
- Classifies based on majority vote of nearest neighbors
- No explicit training phase ("lazy learner")

### 3. Decision Tree
- Tree-based model that makes sequential decisions
- Highly interpretable with clear decision rules
- Can visualize the entire decision process

## Making Predictions
```python
# Example: Predict a new flower species
new_flower = [[5.1, 3.5, 1.4, 0.2]]  # [sepal_l, sepal_w, petal_l, petal_w]
prediction = model.predict(new_flower)
print(f"Predicted species: {iris.target_names[prediction][0]}")
```

# 🎯 Key Features
## Data Exploration

-Statistical summary of features

-Missing value analysis

-Correlation between features

-Species distribution visualization

## Model Training & Evaluation
-Train-test split (80-20)

-Multiple algorithm comparison

-Comprehensive metrics (accuracy, precision, recall, F1-score)

-Confusion matrix analysis

## 🔍 Insights Gained

-Feature Importance: Petal measurements are more discriminative than sepal measurements

-Class Separability: Setosa is easily separable, while Versicolor and Virginica have some overlap

-Model Performance: All models achieve high accuracy (>95%) on this well-structured dataset
