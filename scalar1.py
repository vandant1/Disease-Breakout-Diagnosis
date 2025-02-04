from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle

# Load the actual dataset
diabetes_df = pd.read_csv("C:\\Users\\Hp\\Downloads\\pma Diabetic dataset ex(8)\\diabetes.csv")  # Replace with actual file

# Select only the features used for training
X_train = diabetes_df.drop(columns=["Outcome"])  # Replace "target_column" with actual target column name

# Fit StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

# Save the scaler
with open("diabetes_scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

print("Diabetes scaler saved successfully!")