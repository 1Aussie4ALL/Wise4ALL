from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score
import pickle
import os
import pandas as pd

# Check if the model file exists before loading it
model_filename = 'model.pkl'
if os.path.exists(model_filename):
    with open(model_filename, 'rb') as f:
        clf = pickle.load(f)
else:
    # Load the iris dataset from the provided file instead of sklearn.datasets
    iris_df = pd.read_csv('iris_sample.csv')
    X = iris_df.iloc[:, :-1].values  # All columns except the last one
    y = iris_df.iloc[:, -1].values  # The last column

    # Map species names to numerical labels
    species_to_label = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
    y = [species_to_label[species] for species in y]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define and train the model
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # Save the trained model
    with open(model_filename, 'wb') as f:
        pickle.dump(clf, f)

def predict_flower_species():
    print("Welcome to the Flower Species Prediction Tool!")
    print("Please enter the flower's measurements.")

    # Take user input for the four features (sepal length, sepal width, petal length, petal width)
    try:
        sepal_length = float(input("Enter sepal length (cm): "))
        sepal_width = float(input("Enter sepal width (cm): "))
        petal_length = float(input("Enter petal length (cm): "))
        petal_width = float(input("Enter petal width (cm): "))

        # The model expects a 2D array as input, so wrap the input values in a list of lists
        flower_features = [[sepal_length, sepal_width, petal_length, petal_width]]

        # Use the model to make a prediction
        prediction = clf.predict(flower_features)[0]

        # Map numerical labels to flower species names
        label_to_name = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        predicted_class = label_to_name[prediction]

        print(f"The model predicts that the flower is of the species: {predicted_class}.")
    except ValueError:
        print("Please enter valid numeric values for the flower's measurements.")

while True:
    predict_flower_species()
    choice = input("Do you want to predict another flower? (yes/no): ")
    if choice.lower() != 'yes':
        print("Thank you for using the Flower Species Prediction Tool!")
        break
