from sklearn.preprocessing import StandardScaler
import pandas as pd
import pickle

# Load the actual dataset
heart_df = pd.read_csv("C:\\Users\\Hp\\Downloads\\Heart disease ex(6)\\heart.csv")  # Replace with actual file

# Select only the features used for training
X_train = heart_df.drop(columns=["target"])  # Replace "target_column" with actual target column name

# Fit StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

# Save the scaler
with open("heart_scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

print("Heart scaler saved successfully!")