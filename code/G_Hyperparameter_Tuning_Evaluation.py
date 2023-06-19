### Import packages ################################################################################################

from D_Evaluation_function import *
import pandas as pd
import numpy as np
import dataframe_image as dfi   # to export df as a table

# Sklearn
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif

# Visualization libraries 
import matplotlib.pyplot as plt
from matplotlib import colormaps
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable # used to arrange confusion matrices in subplots

# Import previous data
df_full = pd.read_csv('../data/df_full.csv')
X_train = pd.read_csv('../data/X_train.csv')
X_test = pd.read_csv('../data/X_test.csv')
y_train = pd.read_csv('../data/y_train.csv')
y_test = pd.read_csv('../data/y_test.csv')
best_number_features = [81, 12, 24, 18]
optimal_k = 41

import warnings
warnings.filterwarnings("ignore")


### Fit models on total training set with best hyperparameters (Randomized Search) #############################################################################################

classifiers = ['SVM', 'LR', 'RF', 'KNN']

# Stores the final results 
df_final_performance = pd.DataFrame(columns = ['tp','fp','tn','fn','accuracy', 'precision', 'recall', 'f1', 'roc_auc'] )

# Stores best features
best_features = pd.DataFrame(columns = ['best features'] )

# Stores best hyperparameters
SVM_best_parameters = pd.DataFrame(columns = ['C','kernel','gamma'] )
LR_best_parameters = pd.DataFrame(columns = ['max_iter','penalty','warm_start','solver','C'] )
RF_best_parameters = pd.DataFrame(columns = ['max_leaf_nodes','min_samples_split','max_depth','max_features','n_estimators'] )
KNN_best_parameters = pd.DataFrame(columns = ['n_neighbors','leaf_size','weights','metric'])


# Set hyperparameters search space -> I've chosen range based on internet research ;) -> no parameter is at the very edge of its search range which implies that ranges are adequate
C_range = np.logspace(-10, 10, 21)  # List of C values (Regularization parameter)
gamma_range = np.logspace(-10, 10, 21)  # List of gamma values (Kernel coefficient)
parameters_SVM = {"C": C_range, 
                  "kernel": ['rbf', 'poly'],        # linear kernel takes very long!
                  "gamma": gamma_range.tolist()+['scale', 'auto']}

max_iter = range(100, 500)
penalty = ['l1', 'l2', 'elasticnet']
warm_start = [True, False]
solver = ['lbfgs', 'newton-cg', 'liblinear']
C = np.arange(0, 1, 0.01)
parameters_LR = {'max_iter': max_iter, 
                 'penalty': penalty,
                 'warm_start': warm_start, 
                 'solver': solver, 
                 'C': C}

max_leaf_nodes = [4,5,6,7,8,9]
min_samples_split = [5,10,20,30,40,50]
max_depth = [5,10,15,20]
max_features = [3,4,5]
n_estimators = [30, 50, 100, 200]
parameters_RF = {'max_leaf_nodes': max_leaf_nodes, 
                 'min_samples_split': min_samples_split, 
                 'max_depth': max_depth, 
                 'max_features': max_features,
                 'n_estimators': n_estimators}

neighbors = [optimal_k]
leaf_size = [x for x in range(1,60)]
weights = ['uniform','distance']
metric = ['minkowski','euclidean','manhattan']
parameters_KNN = {'n_neighbors' : neighbors,
       'leaf_size' : leaf_size,
       'weights' : weights,
       'metric' : metric}

# Initialize different models:  -> RandomizedSearch is faster than GrisSearch + we can search in a larger range
clf_SVM = svm.SVC(probability=True, random_state = 1)
clf_LR = LogisticRegression(random_state = 1)
clf_RF = RandomForestClassifier(random_state=1)
clf_KNN = KNeighborsClassifier()

# Auxilary lists
models_raw = [clf_SVM, clf_LR, clf_RF, clf_KNN]
parameters = [parameters_SVM, parameters_LR, parameters_RF, parameters_KNN]
best_params = [SVM_best_parameters, LR_best_parameters, RF_best_parameters, KNN_best_parameters]  # List of all parameter dict names
plt.figure(figsize=(8, 6))  # for roc curve


# Fit + Evaluete each classifier
for i, classif in enumerate(classifiers):
    # Univariate feature selection (with optimal number for each algorithm and on whole training set): 
    UVFS_Selector = SelectKBest(f_classif, k=best_number_features[i])                
    X_train_UVFS = UVFS_Selector.fit_transform(X_train, y_train)      
    X_test_UVFS = UVFS_Selector.transform(X_test)

    # Save best features (utilized features for every algorithm)
    best_features.loc[f'{best_number_features[i]} best features for {classif}:',:] = [UVFS_Selector.get_feature_names_out()]

    # Hyperparameter tuning and model fitting
    model = RandomizedSearchCV(models_raw[i], parameters[i], n_iter=100, scoring = 'accuracy', random_state = 1, verbose = 1, cv=3, n_jobs = -1) # we compare 50 random combinations and take best
    model.fit(X_train_UVFS, y_train.values.ravel())  
    if classif == 'LR':
        clf_LR_RS = model  # we need to save the LR model for future feature importance analyses

    # Evaluation of final performance
    df_final_performance.loc[f'{classif} (test)',:] = eval_Performance(y_test, X_test_UVFS, model, clf_name = classif)
    df_final_performance.loc[f'{classif} (train)',:] = eval_Performance(y_train, X_train_UVFS, model, clf_name = classif)
    
    # Save best hyperparameters
    best_params[i].loc[f'{classif} best parameters:',:] = model.best_params_    
    
    # Plot roc curve
    y_pred_proba = model.predict_proba(X_test_UVFS)[:, 1]   # Predict probabilities for the test set
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba) # Calculate the false positive rate, true positive rate, and thresholds
    auc_score = roc_auc_score(y_test, y_pred_proba)  # Calculate the AUC score
    plt.plot(fpr, tpr, label=f'{classif} (AUC = {auc_score:.2f})')  # Plot the ROC curve
    
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')  # Plot the random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('../output/roc_curve.png')

# Export dataframes to tables (takes some time)
dfi.export(df_final_performance,"../output/df_final_performance.png")  # add dpi = ... for higher resolution output
dfi.export(best_features,"../output/best_features.png")
dfi.export(SVM_best_parameters,"../output/SVM_best_parameters.png")
dfi.export(LR_best_parameters,"../output/LR_best_parameters.png")
dfi.export(RF_best_parameters,"../output/RF_best_parameters.png")
dfi.export(KNN_best_parameters,"../output/KNN_best_parameters.png")



### Visualize confusion matrices #############################################################################################

im = [0,0,0,0]
fig, axes = plt.subplots(2,2, figsize=(10, 10))

for k, classif in enumerate(classifiers):
    axes = axes.flatten()
    tn = df_final_performance.iloc[2*k]['tn']
    fp = df_final_performance.iloc[2*k]['fp']
    fn = df_final_performance.iloc[2*k]['fn']
    tp = df_final_performance.iloc[2*k]['tp']
    confusion_mat = np.array([[tn, fp], [fn, tp]])
    
    im[k] = axes[k].imshow(confusion_mat, cmap='coolwarm', interpolation='nearest')
    axes[k].set_title(f'Confusion Matrix - {classif}')
    tick_marks = np.arange(2)
    axes[k].set_xticks(tick_marks, ['Negative', 'Positive'])
    axes[k].set_yticks(tick_marks, ['Negative', 'Positive'])
    axes[k].set_xlabel('Predicted Label')
    axes[k].set_ylabel('True Label')

    # Add label values inside the plot
    thresh = confusion_mat.max() / 2.
    for i in range(confusion_mat.shape[0]):
        for j in range(confusion_mat.shape[1]):
            axes[k].text(j, i, format(confusion_mat[i, j], 'd'), ha="center", va="center",color="black" if confusion_mat[i, j] > thresh else "black")    
    # Add colorbar
    divider = make_axes_locatable(axes[k])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im[k], cax=cax, orientation='vertical')

plt.tight_layout()  
plt.savefig('../output/confusion_matrices.png')


### Visualize feature importance scores of LR #############################################################################################

coefficients = clf_LR_RS.best_estimator_.coef_
importance = np.abs(coefficients)
importance_scores = (importance / np.sum(importance)).flatten()

fig, ax = plt.subplots(figsize=(12, 6))
plt.bar(np.arange(len(importance_scores)), importance_scores)
ax.set_xticks(np.arange(len(importance_scores)), best_features.iloc[1][0].tolist(), rotation=90)  # set the xticks according to the feature names, and rotate them by 90 degrees
ax.set_title("Normalized feature importance of logistic regression", fontsize=20)
ax.set_ylabel("Normalized absolute model coefficient")
plt.savefig('../output/LR_feature_importance.png')
