import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load your dataset (replace with the actual dataset path)
df = pd.read_csv("C:\\Users\\Hp\\Downloads\\Heart disease ex(6)\\heart.csv")  # Ensure the dataset is correct

# Define features and target variable
X = df.drop(columns=["target"])  # Replace "target" with the actual column name
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model
with open("heart_disease_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Heart Disease Model saved successfully!")
