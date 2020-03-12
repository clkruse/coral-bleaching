import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

file_name = 'processed_window_10_params_11.json'

db = pd.read_json(file_name)

variables = ['sst', 'chlor_a', 'velocity', 'salinity', 'wind_avhrr', 'cloud_transmission']
#variables = ['sst']

#stats = ['mean_', 'med_', 'min_', 'max_']
stats = ['med_']

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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=13)

from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', np.unique(y_train), y_train)
## Algorithm Zoo!

# Decision Tree
from sklearn import tree
from sklearn.tree.export import export_text

percent_positive = sum(y_train)/len(y_train)
percent_negative = 1 - percent_positive

clf_tree = tree.DecisionTreeClassifier(random_state = 42, class_weight = 'balanced', max_depth=5)
clf_tree = clf_tree.fit(x_train, y_train)


preds_test = clf_tree.predict(x_test)
test_rms = np.mean(np.abs(preds_test - y_test))
test_acc = clf_tree.score(x_test, y_test)


preds_train = clf_tree.predict(x_train)
train_rms = np.mean(np.abs(preds_train - y_train))
train_acc = clf_tree.score(x_train, y_train)

print("Test: RMSE - {0:.3f}, Accuracy - {1:.2f}".format(test_rms, test_acc))
print("Train: RMSE - {0:.3f}, Accuracy - {1:.2f}".format(train_rms, train_acc))
print("Chance: RMSE - {0:.3f}, Accuracy - {1:.2f}".format(np.mean(np.abs(y_test - 1)), sum(y_test)/len(y_test)))

for a, b in zip(variables, clf_tree.feature_importances_):
    print("{0}: {1:.2f}".format(a, b))

print(classification_report(y_test, preds_test, target_names=['Healthy', 'Bleaching']))

plt.figure(figsize=[14,8])
tree.plot_tree(clf_tree, feature_names=variables, filled=True, fontsize=8, max_depth=3, rounded=True)
plt.show()


# Random Forest
from sklearn.ensemble import RandomForestClassifier

clf_forest = RandomForestClassifier(max_depth=3, class_weight = 'balanced', n_estimators = 100, random_state = 10)
clf_forest.fit(x_train, y_train)

preds_test = clf_forest.predict(x_test)
test_rms = np.mean(np.abs(preds_test - y_test))
test_acc = clf_forest.score(x_test, y_test)


preds_train = clf_forest.predict(x_train)
train_rms = np.mean(np.abs(preds_train - y_train))
train_acc = clf_forest.score(x_train, y_train)

print("Test: RMSE - {0:.3f}, Accuracy - {1:.2f}".format(test_rms, test_acc))
print("Train: RMSE - {0:.3f}, Accuracy - {1:.2f}".format(train_rms, train_acc))
print("Chance: RMSE - {0:.3f}, Accuracy - {1:.2f}".format(np.mean(np.abs(preds_test - 1)), sum(y_test)/len(y_test)))

for a, b in zip(variables, clf_forest.feature_importances_):
    print("{0}: {1:.2f}".format(a, b))

print(classification_report(y_test, preds_test, target_names=['Healthy', 'Bleaching']))

# Linear Regression
from sklearn.linear_model import LinearRegression

reg = LinearRegression().fit(x_train, y_train)
print(reg.score(x_test, y_test))
preds_test = reg.predict(x_test)

for i in range(10):
    sample = np.random.randint(len(x))
    print("Prediction: {0:.1f}\nActual: {1}".format(np.squeeze(reg.predict(x[sample:sample+1])), y[sample]))
