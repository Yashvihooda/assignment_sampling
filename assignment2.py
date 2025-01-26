
from google.colab import files

# Upload the dataset
uploaded = files.upload()

import pandas as pd

# Load the dataset
data = pd.read_csv("Creditcard_data.csv")
data.head(), data.info(), data.describe()

# Checking the distribution of the target variable (Class)
class_distribution = data['Class'].value_counts(normalize=True) * 100
print(class_distribution)

# Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance the classes
from imblearn.over_sampling import SMOTE
# Separate features and target
X = data.drop(columns=['Class'])
y = data['Class']

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Check the new distribution of the target variable after balancing
balanced_class_distribution = y_balanced.value_counts(normalize=True) * 100
print(balanced_class_distribution)

# Import necessary modules for model training and evaluation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Split the balanced dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced)

# Define custom sampling techniques
def random_sampling(X, y, sample_size):
    indices = np.random.choice(range(len(X)), size=sample_size, replace=False)
    return X.iloc[indices], y.iloc[indices]

def stratified_sampling(X, y, sample_size):
    return train_test_split(X, y, train_size=sample_size, random_state=42, stratify=y)

# Define the sample sizes to be tested
sample_sizes = [100, 200, 300, 400, 500]

# Define the models to be evaluated
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Dictionary to store results for each sampling size and model
evaluation_results = {}

# Loop through each sample size and train the models
for i, sample_size in enumerate(sample_sizes):
    evaluation_results[f"Sample {i + 1}"] = {}

    # Apply random sampling to select a subset of the training data
    X_sample, y_sample = random_sampling(X_train, y_train, sample_size)

    for model_name, model in models.items():
        # Train the model on the sampled data
        model.fit(X_sample, y_sample)

        # Test the model on the test set
        y_pred = model.predict(X_test)

        # Evaluate accuracy of the model
        accuracy = accuracy_score(y_test, y_pred)
        evaluation_results[f"Sample {i + 1}"][model_name] = accuracy

# Convert the results into a DataFrame for easier interpretation
evaluation_results_df = pd.DataFrame(evaluation_results)
print(evaluation_results_df)
