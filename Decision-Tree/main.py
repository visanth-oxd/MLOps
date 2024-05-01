from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd


# Lod th digits datset
digits = datasets.load_digits()
X = digits.data
y = digits.target

# Split the datset inti test and trianing 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

#Initialise the decisoin tree classifier
clf = DecisionTreeClassifier()

# train the model
clf.fit(X_train, y_train)

# make predictions on the testing set
y_pred = clf.predict(X_test)

# Calculate the accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# save the model
joblib.dump(clf, 'treemodel.pkl')



