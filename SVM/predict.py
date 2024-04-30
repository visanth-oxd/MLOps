from joblib import load

# Load the model from the file
clf = load('svmmodel.joblib')

# New data for prediction exmaple
new_data = [[13.71,1.86,2.36,16.6,101,2.61,2.88,0.27,1.69,3.8,1.11,4.0,1035]]  # This is an example. 

# make prediction
prediction = clf.predict(new_data)


print(f" The prediction class for the new data is {prediction}")

