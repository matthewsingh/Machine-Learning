import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
df = pd.read_csv(url)

# Rename columns to match standard names
df.rename(columns={'Siblings/Spouses Aboard': 'SibSp', 'Parents/Children Aboard': 'Parch'}, inplace=True)

# Display first 5 rows
print(df.head())

#### Data Cleaning - Handle Missing Values ####
# Check for missing values
print(df.isnull().sum())

# Fill missing Age with median age
df['Age'] = df['Age'].fillna(df['Age'].median())

# Feature engineering
# Family size
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

# Is alone
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

# Age bins
df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, 80], labels=['Child', 'Teen', 'Adult', 'Middle', 'Senior'])

# Fare bins
df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=['Low', 'Mid', 'High', 'VeryHigh'])

# Convert categorical to numeric
df['AgeBin'] = df['AgeBin'].cat.codes
df['FareBin'] = df['FareBin'].cat.codes

# Convert Sex to numeric (male=0, female=1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Select relevant features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone', 'AgeBin', 'FareBin']
X = df[features]
y = df['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#### Test Multiple Models ####
# Define models to test
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'SVM': SVC(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# Dictionary to store results
results = {}

for name, model in models.items():
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {'accuracy': accuracy, 'predictions': y_pred}
    print(f"{name} Accuracy: {accuracy:.2f}")

# Plot accuracy comparison
model_names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in model_names]
plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies, color='skyblue')
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f'{acc:.2f}', ha='center')
plt.savefig('model_comparison.png')
plt.show()

# Confusion matrix for the best model (Random Forest as example)
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = models[best_model_name]
y_pred_best = results[best_model_name]['predictions']
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix for {best_model_name}')
plt.savefig('confusion_matrix.png')
plt.show()

# Plot predictions vs actual for the best model
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual', marker='o')
plt.plot(y_pred_best, label='Predicted', marker='x')
plt.title(f'Predicted vs Actual Survival ({best_model_name})')
plt.xlabel('Sample Index')
plt.ylabel('Survival')
plt.legend()
plt.savefig('predictions_vs_actual.png')
plt.show()

# Check feature importance for tree-based models
tree_models = ['Decision Tree', 'Random Forest']
for name in tree_models:
    if name in models:
        model = models[name]
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        print(f"\n{name} Feature Importance:")
        print(feature_importance)
