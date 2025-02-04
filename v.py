import pickle
with open("C:\\Users\\Hp\\Downloads\\diabetes_model.pkl", "rb") as f:
    try:
        model = pickle.load(f)
        print("Model loaded successfully")
    except Exception as e:
        print("Error loading model:", e)
