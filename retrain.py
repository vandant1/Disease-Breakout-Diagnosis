import pickle
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
# Retrain your model here

with open("diabetes_model.pkl", "wb") as file:
    pickle.dump(model, file)
