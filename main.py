# https://towardsdatascience.com/introduction-to-shap-with-python-d27edc23c454
# https://github.com/conorosully/medium-articles/blob/master/src/shap_tutorial.ipynb
# https://archive.ics.uci.edu/ml/datasets/Abalone

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import accuracy_score, confusion_matrix
import shap

# import dataset
data = pd.read_csv('abalone.data', names=['sex', 'length', 'diameter', 'height', 'whole weight', 'shucked weight', 'viscera weight', 'shell weight', 'rings'])

y = data['rings']
X = data.drop('rings', axis=1)

print(f'{len(data)} observations')
data.head()

plt.figure(figsize=(18, 9))
# Plot 1
plt.subplot(121)
plt.scatter(data['shucked weight'], data['rings'])
plt.ylabel('Rings', size=20)
plt.xlabel('Shucked weight', size=20)

# Plot 2
plt.subplot(122)
plt.boxplot(data[data.sex == 'I']['rings'], positions=[1])
plt.boxplot(data[data.sex == 'M']['rings'], positions=[2])
plt.boxplot(data[data.sex == 'F']['rings'], positions=[3])
plt.xticks(ticks=[1, 2, 3], labels=['I', 'M', 'F'], size=15)
plt.xlabel('Sex', size=20)

plt.savefig(f'data_exploration.png')

# Create dummy variables
X['sex.M'] = [1 if s == 'M' else 0 for s in X['sex']]
X['sex.F'] = [1 if s == 'F' else 0 for s in X['sex']]
X['sex.I'] = [1 if s == 'I' else 0 for s in X['sex']]
X = X.drop('sex', axis=1)

X.head()

# Train model
model = xgb.XGBRegressor(objective='reg:squarederror')
model.fit(X, y)

# Get predictions
y_pred = model.predict(X)

# Model evaluation
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
plt.scatter(y, y_pred)
plt.plot([0, 30], [0, 30], color='r', linestyle='-', linewidth=2)
plt.ylabel('Predicted', size=20)
plt.xlabel('Actual', size=20)

plt.savefig(f'regression_evaluation.png')

# Get shap values
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Waterfall plot for first observation
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
shap.plots.waterfall(shap_values[0])
plt.savefig(f'waterfall_obs0.png')

# Force plot for first observation
# todo this doesnt appear to work.
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
shap.plots.force(shap_values[0])
plt.savefig(f'force_obs0.png')







shap.plots.force(shap_values[0:100])

# Mean SHAP
shap.plots.bar(shap_values, show=False)

plt.savefig(f'mean_shap.png')

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
plot_importance(model, ax=ax, grid=False, height=0.5)

plt.title('')
plt.ylabel('')
plt.xlabel('F score', size=20)

plt.savefig(path.format('Feature_importance.png'))

# Get expected value and shap values array
expected_value = explainer.expected_value
shap_array = explainer.shap_values(X)

# Descion plot for first 10 observations
shap.decision_plot(expected_value, shap_array[0:10], feature_names=list(X.columns), show=False)

plt.savefig(f'decision_plot.png')

# Beeswarm plot
shap.plots.beeswarm(shap_values, show=False)

plt.savefig(f'beeswarm.png')

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

shap.plots.scatter(shap_values[:, 'shell weight'], ax=ax[0], show=False)
shap.plots.scatter(shap_values[:, 'shucked weight'], ax=ax[1], show=False)

plt.savefig(f'shap_scatter.png')

shap.plots.scatter(shap_values[:, 'shucked weight'], color=shap_values[:, 'shell weight'])

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))

plt.scatter(data['shell weight'], data['shucked weight'], c=data['rings'], cmap='bwr')
plt.colorbar(label='Number of Rings', orientation='vertical')

plt.xlabel('shucked weight', size=20)
plt.ylabel('shell weight', size=20)

plt.savefig(f'weight_interaction.png')

# Binary target varibale
y = [1 if y_ > 10 else 0 for y_ in y]

# Train model
model = xgb.XGBClassifier(objective='binary:logistic')
model.fit(X, y)

y_pred = model.predict(X)

print(confusion_matrix(y, y_pred))
accuracy_score(y, y_pred)

# Get shap values
explainer = shap.Explainer(model)
shap_values = explainer(X)

# waterfall plot for first observation
shap.plots.waterfall(shap_values[0])

shap.plots.beeswarm(shap_values)
