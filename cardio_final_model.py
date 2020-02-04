############################################################
################ Multi-class Classification ################ 
############################################################


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier


#---------------------- Import data ------------------------

path_data = "/cardio/"
path_charts = "/cardio/charts/"

dataset = pd.read_csv(path_data + "cardiotoc.csv")


#------------ Final Model -------------

# Parameters
no_trees = 50
no_layers = 2
learning_rate = 0.05
subsample_lvl = 0.5
early_stopping = 60

#  Prepare data
X = np.array(dataset.iloc[:,:-1])
Y = np.array(dataset.iloc[:,-1:])

scaler = StandardScaler()
pca = PCA(0.95)

scaler.fit(X)

X = scaler.transform(X)
pca.fit(X)
X = pca.transform(X)                   

model = XGBClassifier(n_estimators= no_trees, 
                      max_depth = no_layers , 
                      early_stopping_rounds=early_stopping, 
                      learning_rate = learning_rate, 
                      colsample_bylevel= subsample_lvl, 
                      objective = "softmax", 
                      random_state = 42)

model.fit(X, Y, verbose = False)

#---- Feature Importance ---

from xgboost import plot_importance, plot_tree
ax = plot_importance(model)
ax.figure.savefig(path_charts + 'Feature_Importance.png', dpi = 300, bbox_inches = "tight")

#---- Display Tree -----

plt.rcParams['figure.figsize'] = [20,20]
plot_tree(model, num_trees=model.get_booster().best_iteration)
plt.savefig(path_charts + 'Tree.png',  dpi = 300, bbox_inches = "tight")

