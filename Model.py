# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset and model selection
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Evaluation metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#### 3. **Load the Dataset**The Iris dataset is built into `scikit-learn`, making it easy to load:```python
# Load the dataset
iris = load_iris()

# Create a DataFrame for better visualization
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = df['target'].apply(lambda x: iris.target_names[x])

# Display the first 5 rows
print(df.head())

#### 4. **Explore the Data (EDA)** Let's understand the dataset better:```python
# Basic information
print(df.info())

# Statistical summary
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Visualize the data
sns.pairplot(df, hue='species', palette='viridis')
plt.show()

#### 5. **Prepare the Data for Training** We need to separate our features (X) and target variable (y), then split into training and testing sets:```python
# Features (X) and target (y)
X = df.drop(['target', 'species'], axis=1)
y = df['target']

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

#### 6. **Choose and Train a Model** Let's start with a simple Logistic Regression model:```python
# Initialize the model
model = LogisticRegression(max_iter=200)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

#### 7. **Evaluate the Model** Let's see how well our model performs:```python
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#### 8. **Try Different Models** Let's experiment with other algorithms to see which performs best:```python
# List of models to try
models = {
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression(max_iter=200)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.3f}")

# Compare results
print("\nModel Comparison:")
for name, accuracy in results.items():
    print(f"{name}: {accuracy:.3f}")

#### 10. **Visualize the Decision Tree Model** Let's visualize the decision tree to understand how it makes decisions:```python
# Train a Decision Tree model for visualization
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Plot the decision tree
plt.figure(figsize=(12, 8))
plot_tree(dt_model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
plt.title('Decision Tree Visualization')
plt.show()

#### 9. **Visualize Logistic Regression Decision Boundaries** To visualize how logistic regression separates the classes, we can plot the decision boundaries using two features. Let's use petal length and petal width for clarity:```python
# Select two features for visualization (petal length and petal width)
X_vis = X[['petal length (cm)', 'petal width (cm)']].values
y_vis = y.values

# Train logistic regression on these two features
model_vis = LogisticRegression(max_iter=200)
model_vis.fit(X_vis, y_vis)

# Create a mesh grid for plotting decision boundaries
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict on the mesh grid
Z = model_vis.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundaries
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, cmap='viridis', edgecolor='k', s=50)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Logistic Regression Decision Boundaries (Petal Length vs Petal Width)')
plt.colorbar(scatter, ticks=[0, 1, 2], label='Species')
plt.clim(-0.5, 2.5)
plt.show()

#### 10. **Make a Prediction on New Data** Let's use our trained model to predict a new flower's species:```python
# Example new flower measurements [sepal length, sepal width, petal length, petal width]
new_flower = [[5.1, 3.5, 1.4, 0.2]] # Example measurements

# Predict the species
prediction = model.predict(new_flower)
predicted_species = iris.target_names[prediction][0]

print(f"The predicted species is: {predicted_species}")
