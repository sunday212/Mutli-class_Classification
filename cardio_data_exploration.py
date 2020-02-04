############################################################
################ Multi-class Classification ################ 
############################################################

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

#---------------------- Import data ------------------------

path_data = "/Users/lisa/Desktop/blue_sky/My Learnings/Machine Learning/Mastering Machine Learning/Projects/01_Classification - Multiclass/cardio/"
path_charts = "/Users/lisa/Desktop/blue_sky/My Learnings/Machine Learning/Mastering Machine Learning/Projects/01_Classification - Multiclass/cardio/charts/"

dataset = pd.read_csv(path_data + "cardiotoc.csv")


#------------------ Explore the data -----------------------

dataset.describe()                          # Statistics
dataset.isna().sum()                        # Check number of NAs

sns.boxplot(data=dataset.iloc[:,0:2])       # Check outliers      
sns.boxplot(data=dataset.iloc[:,2:16])
sns.boxplot(data=dataset.iloc[:,16:24])
sns.boxplot(data=dataset.iloc[:,24:len(dataset)])

# Distribution of target variable
sns.barplot(x= dataset['NSP'].value_counts().index , y =dataset['NSP'].value_counts()/len(dataset))
plt.ylabel('Proportional split')
plt.xlabel("Target labels")
plt.savefig(path_charts + 'Target Class Distribution.png', dpi = 1000, bbox_inches = "tight")


# Freuency charts
no_cols = len(dataset.columns)
for i in range(no_cols):
    plt.hist(dataset.iloc[:,i:(i+1)].values,15)
    plt.title("Frequency plot of " + dataset.columns[i])
    plt.savefig(path_charts + "Histogram_" + dataset.columns[i] + ".png")
    plt.close()


# Visualise Correlation Matrix
features = dataset.iloc[:,:-1]
correlations = features.corr()

f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(correlations, cmap=cmap, vmax=.3, center=0,
square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title("Correlation Matrix of Features")
plt.savefig(path_charts + 'Correlation_Matrix_of_Features.png', dpi = 1000, bbox_inches = "tight")

