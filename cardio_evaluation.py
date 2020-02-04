############################################################
################ Multi-class Classification ################ 
############################################################

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score 
from xgboost import XGBClassifier
import statistics as st
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline


#---------------------- Import data ------------------------

path_data = "/Users/lisa/Desktop/blue_sky/My Learnings/Machine Learning/Mastering Machine Learning/Projects/01_Classification - Multiclass/cardio/"
path_charts = "/Users/lisa/Desktop/blue_sky/My Learnings/Machine Learning/Mastering Machine Learning/Projects/01_Classification - Multiclass/cardio/charts/"

dataset = pd.read_csv(path_data + "cardiotoc.csv")


#-------------------- Prepare inputs for model -----------------

X = np.array(dataset.iloc[:,:-1])
Y = np.array(dataset.iloc[:,-1:])

# scalar to transform data
scaler = StandardScaler()

# Finding minimum number of principal components such that 95% of the variance is retained
pca = PCA(0.95)

pca.fit(X)
pca_X = pca.transform(X)                    # Transform to PCAs

# Evaluate using stratified k-fold cross-validation
splits = 10
skfold = StratifiedKFold(n_splits=splits, random_state=100)

# Target class weights for cost minimisation
class_weights = class_weight.compute_class_weight('balanced',np.unique(dataset['NSP']),dataset['NSP'])



#------------------- Tune Parameters for XGBoost model  -----------------------

# ------ Find early stopping point ------

i = -1
for train_index, test_index in skfold.split(X, Y):
    train_X, test_X = X[train_index], X[test_index]
    train_Y, test_Y = Y[train_index], Y[test_index]
    
    scaler.fit(train_X)

    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)
         
    pca.fit(train_X)
    
    train_X = pca.transform(train_X)                    # Transform to PCAs
    test_X = pca.transform(test_X)

    eval_set = [(train_X, train_Y), (test_X, test_Y)]                       # evaluation set for early stopping     

    # Define the model
    model = XGBClassifier(objective = "softmax", class_weight = class_weights, random_state = 42)

    # Oversample the minority classes with SMOTE, undersampling the majority class with Tomek Links
    resample = SMOTETomek(tomek = TomekLinks(sampling_strategy='not majority'))
    pipeline = Pipeline(steps = [('r', resample), ('m', model)])

    # Fit the model on the data
    pipeline.fit(train_X, train_Y, m__eval_metric = 'mlogloss', m__eval_set = eval_set,  m__verbose = False)
    
    # Make predictions on the test data
    predictions = pipeline.predict(test_X)

    # Find early stopping epoch no
    eval_results = model.evals_result()
    epochs = len(eval_results['validation_0']['mlogloss'])
    x_axis = range(0, epochs)

    # plot log loss
    fig, ax = plt.subplots()  
    ax.plot(x_axis,eval_results['validation_0']['mlogloss'], label = 'Train', color = 'blue' )
    ax.plot(x_axis,eval_results['validation_1']['mlogloss'], label = 'Test', color ='orange' )
    ax.legend()
    plt.ylabel('Log Loss')
    plt.xlabel('Epochs')
    plt.title('Epochs vs Log Loss')
    i = i + 1
    plt.savefig(path_charts + "Epochs_" + str(i+1) + ".png", dpi = 300, bbox_inches = "tight")


# Optimal no epochs = 60
    

#---------- Prepare Model --------

model = XGBClassifier(objective = "softmax", class_weight = class_weights, random_state = 42)

# oversample the minority classes with SMOTE, undersampling the majority class with Tomek Links
resample = SMOTETomek(tomek = TomekLinks(sampling_strategy='not majority'))
pipeline = Pipeline(steps = [('r', resample), ('m', model)])


#------------Find optimal number of layers in trees and number of trees together  ---------

n_estimators = [50, 75, 100]
max_depth = [2, 3, 5, 7]

# sorted(pipeline.get_params().keys())   # key param keys

params =  { 'm__n_estimators': n_estimators,
            'm__max_depth': max_depth}

grid_search = GridSearchCV(pipeline, params, scoring = "neg_log_loss", cv = skfold, verbose = 0)
grid_result = grid_search.fit(pca_X, Y)

# summarise results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, params))

scores = np.array(means).reshape(len(max_depth), len(n_estimators))
for i, value in enumerate(max_depth):
    plt.plot(n_estimators, scores[i], label='depth: ' + str(value))
    plt.legend()
    plt.title("Log loss for no. of trees and depth of trees")
    plt.xlabel('n_estimators')
    plt.ylabel('Log Loss')

plt.savefig(path_charts + 'No_trees_vs_max_layers.png', dpi = 1000, bbox_inches = "tight")

# Optimal number of layers: 2
# Optimal number of trees: 50



#--------- Find optimal learning rate and number of trees ---------

# Grid search    

learning_rate = [0.001, 0.01, 0.05, 0.1, 0.2]
n_estimators = [50, 100, 150, 200]

params =  { 'm__learning_rate': learning_rate,
           'm__n_estimators': n_estimators }

grid_search = GridSearchCV(pipeline, params, scoring = "neg_log_loss", cv = skfold, verbose = 0)
grid_result = grid_search.fit(pca_X, Y)

# summarise results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, params))

scores = np.array(means).reshape(len(learning_rate), len(n_estimators))
for i, value in enumerate(learning_rate):
    plt.plot(n_estimators, scores[i], label='learning_rate: ' + str(value))
    plt.legend()
    plt.xlabel('No of trees')
    plt.ylabel('Log Loss')

plt.savefig(path_charts + 'No_trees_vs_learning_rate.png', dpi = 1000, bbox_inches = "tight")

# Optimal learning rate: 0.05
# Optimal no of trees: 150


#--------- Find optimal sub-sample by rows ---------

subsample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
params =  { 'm__subsample': subsample }

grid_search = GridSearchCV(pipeline, params, scoring = "neg_log_loss", cv = skfold, verbose = 0)
grid_result = grid_search.fit(pca_X, Y)

# summarise results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, params))

plt.errorbar(subsample, means, yerr=stds)
plt.title("Sub sample by rows vs Log Loss")
plt.xlabel('Sub sample by rows')
plt.ylabel('Log Loss')
plt.savefig(path_charts + 'Subsampling_rows.png', dpi = 1000, bbox_inches = "tight")

# Optimal subsample: 0.8 
# Log loss: -0.932


#--------- Find optimal sub-sample by columns ---------

subsample_cols = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
params =  { 'm__subsample_cols': subsample_cols }

grid_search = GridSearchCV(pipeline, params, scoring = "neg_log_loss", cv = skfold, verbose = 0)
grid_result = grid_search.fit(pca_X, Y)

# summarise results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, params))

plt.errorbar(subsample, means, yerr=stds)
plt.title("Sub sample by columns vs Log Loss")
plt.xlabel('Sub sample by columns')
plt.ylabel('Log Loss')
plt.savefig(path_charts + 'Subsampling_columns.png', dpi = 1000, bbox_inches = "tight")

# Optimal subsample: 0.1 
# Log loss: -0.938


#--------- Find optimal sub-sample by level ---------

subsample_lvl = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
params =  { 'm__subsample_lvl': subsample_lvl }

grid_search = GridSearchCV(pipeline, params, scoring = "neg_log_loss", cv = skfold, verbose = 0)
grid_result = grid_search.fit(pca_X, Y)

# summarise results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, params))

plt.errorbar(subsample, means, yerr=stds)
plt.title("Sub sample by levels vs Log Loss")
plt.xlabel('Sub sample by levels')
plt.ylabel('Log Loss')
plt.savefig(path_charts + 'Subsampling_level.png', dpi = 1000, bbox_inches = "tight")

# Optimal subsample: 0.5 
# Log loss: -0.933


# Tune using level subsamping



#-------------------------- Final model evaluation -----------------------

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

# Evaluate using stratified k-fold cross-validation
splits = 10
skfold = StratifiedKFold(n_splits=splits, random_state=100)

accuracy_scores= []
confn_matrices=np.zeros((splits,len(np.unique(Y)),len(np.unique(Y))))
evaluation = []
precisions = []
recalls = []
f1s= []

i = -1
for train_index, test_index in skfold.split(X, Y):
    train_X, test_X = X[train_index], X[test_index]
    train_Y, test_Y = Y[train_index], Y[test_index]
    
    scaler.fit(train_X)

    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)
    
    pca.fit(train_X)
    
    train_X = pca.transform(train_X)                   
    test_X = pca.transform(test_X)
    
    eval_set = [(test_X, test_Y)]                       
    
    # Fit the model on the data
    model = XGBClassifier(n_estimators= no_trees, 
                          max_depth = no_layers , 
                          early_stopping_rounds=early_stopping, 
                          learning_rate = learning_rate, 
                          colsample_bylevel= subsample_lvl, 
                          objective = "softmax", 
                          random_state = 42)
    
    model.fit(train_X, train_Y,  eval_metric = 'mlogloss', eval_set = eval_set, verbose = False)

    # Make predictions on the test data
    predictions = model.predict(test_X)

    # Evaluate performance of the predictions
    accuracy = accuracy_score(test_Y, predictions)
    confn_matrix = confusion_matrix(test_Y, predictions)
    
    accuracy_scores.append(accuracy)
    accuracy_avg = st.mean(accuracy_scores)
    
    i = i +1 
    confn_matrices[i] = confn_matrix   
    confusionmatrix_avg =confn_matrices.mean(axis = 0)     
    
    eval_results = model.evals_result()
    eval = eval_results['validation_0']['mlogloss'][-1]
    evaluation.append(eval)
    evaluation_avg = st.mean(evaluation)

    precision = precision_score(test_Y, predictions, labels = [2,3], average='micro')
    recall = recall_score(test_Y, predictions, average='micro')
    f1 = f1_score(test_Y, predictions, average='micro')
    
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    
    precision_avg = st.mean(precisions)
    recall_avg = st.mean(recalls)
    f1_avg = st.mean(f1s)    
    

print('Precision: %.3f' % precision_avg)         # The proportion of obervations that were correctly assined the positive class
print('Recall (Sensitivity): %.3f' % recall_avg) # How well the positive classes were predicted
print('F1 Score: %.3f' % f1_avg)
print('Accuracy: %.3f' % accuracy_avg)
print('Log loss: %.3f' % evaluation_avg)
print(np.round(confusionmatrix_avg,0).astype(int))


#-- Final evaluation results --

# Precision: 0.981
# Recall: 0.977
# F1 Score: 0.977
# Accuracy: 0.977
# Log loss: 0.192

# Confusion Matrix
#
#                 predicted
#                 cl1  cl2  cl3
# actual class 1 [164   0   0]
#        class 2 [  3  26   0]
#        class 3 [  1   0  16]



