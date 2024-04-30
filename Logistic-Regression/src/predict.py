import joblib 

# Load model from file
model = joblib.load('lrmodel.pkl')

# New data for prediction - exmaple
new_data = [[5.1,3.5.1.4,0.2]]

# Use model to predict the class of the data
prediction = model.predict(new_data)

print(f"Prediction : {prediction}")

ß