############################################################
################ Multi-class Classification ################ 
############################################################

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


#---------------------- Import data ------------------------

path_data = "/Users/lisa/Desktop/blue_sky/My Learnings/Machine Learning/Mastering Machine Learning/Projects/01_Classification - Multiclass/cardio/"
path_charts = "/Users/lisa/Desktop/blue_sky/My Learnings/Machine Learning/Mastering Machine Learning/Projects/01_Classification - Multiclass/cardio/charts/"

dataset = pd.read_csv(path_data + "cardiotoc.csv")


#------ Prepare  data ----------

X = np.array(dataset.iloc[:,:-1])

scaler = StandardScaler()
scaler.fit(X)

X = scaler.transform(X)
X= scaler.transform(X)

Y = dataset["NSP"]

#------------------- Find the principal components -----------------------

# Finding minimum number of principal components such that 95% of the variance is retained
pca = PCA(0.95)
pca.fit(X)                                          # Fit PCA on the training set

pca.n_components_                                   # No. of PCAs chosen after fitting 


#------------------- View and visualise principal components -------------

principal_components = pca.fit_transform(X)   # Transform to PCAs

# Table of principle components
pcs = list(range(1,pca.n_components_ +1))
pcs = ["pc" + str(x) for x in pcs]

principal_table = pd.DataFrame(principal_components, columns = pcs)
principal_table = pd.concat([principal_table,  Y.reset_index(drop=True)], axis = 1)


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel(principal_table.columns[0], fontsize = 15)
ax.set_ylabel(principal_table.columns[1], fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
ax.set_ylim(-4,0)
ax.set_xlim(-1.5,0.5)
targets = Y.unique()
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = principal_table['NSP'] == target
    ax.scatter(principal_table.loc[indicesToKeep, principal_table.columns[0]]
               , principal_table.loc[indicesToKeep, principal_table.columns[1]]
               , c = color
               , s = 10)
ax.legend(targets)
ax.grid()
plt.savefig(path_charts + 'Principal_Component_Analysis.png', dpi = 1000)


