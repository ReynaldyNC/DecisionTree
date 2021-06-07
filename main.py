import pandas as pd

from sklearn.tree import DecisionTreeClassifier

# Load iris dataset
iris = pd.read_csv('data/iris.csv')

# Remove unimportant column
iris.drop('Id', axis=1, inplace=True)

# Separate attribute and label
x = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris['Species']

# Create DecisionTree model
tree_model = DecisionTreeClassifier()

# Train data
tree_model.fit(x, y)

# Model prediction
model = tree_model.predict([[6.2, 3.4, 5.4, 2.3]])

print(model)
