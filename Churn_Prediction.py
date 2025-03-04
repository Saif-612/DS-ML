import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Create mock customer behavior data
np.random.seed(42)

# Generate demographic data
num_customers = 1000
customer_ids = np.arange(1, num_customers + 1)
age = np.random.randint(18, 70, size=num_customers)
gender = np.random.choice(['Male', 'Female'], size=num_customers)
income = np.random.randint(20000, 150000, size=num_customers)

# Generate purchase history data
avg_purchase_value = np.random.uniform(20, 500, size=num_customers)
purchase_frequency = np.random.randint(1, 20, size=num_customers)
total_spent = avg_purchase_value * purchase_frequency

# Generate engagement data
days_since_last_purchase = np.random.randint(0, 365, size=num_customers)
customer_support_calls = np.random.randint(0, 10, size=num_customers)
website_visits_last_month = np.random.randint(0, 30, size=num_customers)

# Target: Churn (1 = churned, 0 = retained)
churn = np.random.choice([0, 1], size=num_customers, p=[0.7, 0.3])

# Create DataFrame
data = pd.DataFrame({
    'CustomerID': customer_ids,
    'Age': age,
    'Gender': gender,
    'Income': income,
    'AvgPurchaseValue': avg_purchase_value,
    'PurchaseFrequency': purchase_frequency,
    'TotalSpent': total_spent,
    'DaysSinceLastPurchase': days_since_last_purchase,
    'CustomerSupportCalls': customer_support_calls,
    'WebsiteVisitsLastMonth': website_visits_last_month,
    'Churn': churn
})

# Print dataset preview
print("\nFirst 5 rows of the dataset:")
print(data.head())

# Print dataset info
print("\nDataset Summary:")
print(data.info())

# Print statistics of the dataset
print("\nDescriptive Statistics:")
print(data.describe())

# Step 2: Preprocessing

# Encode categorical variables (Gender)
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

# Select features and target
X = data.drop(columns=['CustomerID', 'Churn'])  # Drop CustomerID as it's not a feature
y = data['Churn']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Print dataset split information
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Step 3: Train the model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = rf_classifier.predict(X_test)

# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print model performance metrics
print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", class_report)
print("\nConfusion Matrix:\n", conf_matrix)