import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split

file_name = 'processed_window_10_params_11.json'

db = pd.read_json(file_name)

variables = ['sst', 'chlor_a', 'velocity', 'salinity', 'wind_avhrr', 'cloud_transmission']
#variables = ['sst']

stats = ['mean_', 'med_', 'min_', 'max_']
stats = ['mean_']

features = []

for stat in stats:
    for variable in variables:
        if stat + variable in db.columns:
            features.append(db[stat + variable])

features = np.array(features).T
labels = np.array(db['severity'])

features_filtered = []
labels_filtered = []
for entry, label in zip(features, labels):
    if sum(np.isnan(entry)) == 0:
        features_filtered.append(entry)
        labels_filtered.append(label)

labels_filtered = np.array(labels_filtered)

classes = labels_filtered != 0
#y = np.array(labels_filtered)
x = np.array(features_filtered)
y = np.array(classes).astype(int)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=10)


## Algorithm Zoo!

# Decision Tree
from sklearn import tree
from sklearn.tree.export import export_text

clf_tree = tree.DecisionTreeClassifier(random_state = 42)
clf_tree = clf_tree.fit(x_train, y_train)


preds_test = clf_tree.predict(x_test)
test_rms = np.mean(np.abs(preds_test - y_test))
test_acc = clf_tree.score(x_test, y_test)


preds_train = clf_tree.predict(x_train)
train_rms = np.mean(np.abs(preds_train - y_train))
train_acc = clf_tree.score(x_train, y_train)

print("Test: RMSE - {0:.3f}, Accuracy - {1:.2f}".format(test_rms, test_acc))
print("Train: RMSE - {0:.3f}, Accuracy - {1:.2f}".format(train_rms, train_acc))
print("Chance: RMSE - {0:.3f}, Accuracy - {1:.2f}".format(np.mean(np.abs(preds_test - 1)), sum(preds_test)/len(preds_test)))

for a, b in zip(variables, clf_tree.feature_importances_):
    print(a, b)


# Linear Regression
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(x, y)
reg.score(x, y)

for i in range(10):
    sample = np.random.randint(len(x))
    print("Prediction: {0:.1f}\nActual: {1}".format(np.squeeze(reg.predict(x[sample:sample+1])), y[sample]))
