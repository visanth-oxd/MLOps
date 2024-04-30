
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib


def load_data():
    # Load the data
    data = pd.read_csv('iris.csv', header=None)
    data.columns = ['sepal_length','sepal_width','petal_lngth','petal_width','class']
    return data

def preprocess_data(data):
    # Preprocess thedata
    le = LabelEncoder()
    data['class'] = le.fit_transform(data['class'])
    return data

def train_model(X_train, y_train):
    #Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model
    
    
def evaluate_model(model, X_test, y_test):
    # Evluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")    
    
    
def main():
    data = load_data()
    data = preprocess_data(data)
    X =data.drop('class',axis=1)
    y= data['class']
    X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)    
    
    # Train and evaluate the model
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test,y_test)
    
    # Save the model
    joblib.dump(model, 'lrmodel.pkl')
    
if __name__ == "__main__":
    main()    