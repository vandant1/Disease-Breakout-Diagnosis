# Step 1: Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
#from google.colab import files

# Step 2: Load the Dataset
#uploaded = files.upload()  # Upload the dataset
file_path = "C:\\Users\\Hp\\Downloads\\pma Diabetic dataset ex(8)\\diabetes.csv"  
df = pd.read_csv(file_path)

# Step 3: Explore the Dataset
print("Dataset Shape:", df.shape)
print(df.head())
print("\nSummary Statistics:\n", df.describe())
print("\nMissing Values:\n", df.isnull().sum())

# Step 4: Handle Missing/Zero Values Properly
zero_replacement_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_replacement_cols] = df[zero_replacement_cols].replace(0, np.nan)  # Replace only in selected features
df.fillna(df.median(), inplace=True)  # Fill NaNs with median values

# Step 5: Feature Selection and Target Variable
X = df.iloc[:, :-1]  
y = df.iloc[:, -1]  

# Step 6: Standardizing the Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Splitting the Data (Stratified to Balance Target Classes)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training Samples: {X_train.shape[0]}, Testing Samples: {X_test.shape[0]}")

# Step 8: Choosing and Training the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 9: Evaluating the Model
y_pred = model.predict(X_test)

print("\nAccuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 10: Save the Model and Scaler
joblib.dump(model, "C:\\Users\\Hp\\Downloads\\pma Diabetic dataset ex(8)")
joblib.dump(scaler, "C:\\Users\\Hp\\Downloads\\pma Diabetic dataset ex(8)")

print("\n✅ Model and Scaler Saved Successfully!")

