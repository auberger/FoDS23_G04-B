### Import packages ################################################################################################

from D_Evaluation_function import *
import pandas as pd
import numpy as np

# Sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_classif

# Visualization libraries 
import matplotlib.pyplot as plt
from matplotlib import colormaps
import seaborn as sns

# Import previous data
df_full = pd.read_csv('../data/df_full.csv')
X_train = pd.read_csv('../data/X_train.csv')
y_train = pd.read_csv('../data/y_train.csv')

import warnings
warnings.filterwarnings("ignore")


### Feature selection with wrapper method (find optimal number of features for each algorithm) ####################################################################################
### NOTE: THIS CODE TAKES ~20 MIN! -> reduce the steps (last number in line 25) for faster execution ####################################################################################

variables_numbers = [x for x in range(1, len(X_train.columns)+1, 1)]  # we are basically trying out all possible numbers of features (decided with UVFS) for each algorithm
kf = KFold(n_splits=5, shuffle=True, random_state = 1)   # no stratified CV needed since we have perfectly balanced classes
num_columns = X_train._get_numeric_data().columns
classifiers = ['SVM', 'LR', 'RF', 'KNN']

# Stores the temporarily results for each iteration of hyperparameter tuning
df_evaluation = pd.DataFrame(columns = ['tp','fp','tn','fn','accuracy', 'precision', 'recall', 'f1', 'roc_auc'] )

# Stores best features
best_features_in_selection = pd.DataFrame(columns = ['best features'] )

# Stores the average results and std
df_mean_performance = pd.DataFrame(columns = ['tp','fp','tn','fn','accuracy', 'precision', 'recall', 'f1', 'roc_auc'] )
df_std_performance = pd.DataFrame(columns = ['tp','fp','tn','fn','accuracy', 'precision', 'recall', 'f1', 'roc_auc'] )


for var_num in variables_numbers:
    fold = 1
    for train_index, val_index in kf.split(X_train): # We do the 5-fold CV on the training set only, since the hyperparameter tuning should leave out the test data!
        
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        
        # Standardize the numerical features 
        sc = StandardScaler()
        X_train_fold[num_columns] = sc.fit_transform(X_train_fold[num_columns])
        X_val_fold[num_columns]  = sc.transform(X_val_fold[num_columns])
        
        # Univariate feature selection:  -> filter method = selecting features before fitting models (comp. less expensive then e.g. wrapper methods)
        UVFS_Selector = SelectKBest(f_classif, k=var_num)                # Select top features
        X_UVFS = UVFS_Selector.fit_transform(X_train_fold, y_train_fold) # ...but only on training data!
        X_UVFS_val = UVFS_Selector.transform(X_val_fold)
        
        # Save best features
        best_features_in_selection.loc[f'Best {var_num} features for fold {fold}',:] = [UVFS_Selector.get_feature_names_out()]
        
        # Initialize different models:  -> we just train the classifiers with default settings (could be limitation but reduces time)
        clf_SVM = svm.SVC(C=1, kernel='rbf', gamma='auto', probability=True, random_state = 1)
        clf_LR = LogisticRegression(max_iter=296, penalty='l2', warm_start=True, solver='liblinear', C=0.6, random_state = 1)
        clf_RF = RandomForestClassifier(max_leaf_nodes=8, min_samples_split=10, max_depth=15, max_features=5, n_estimators=30, random_state=1)        
        clf_KNN = KNeighborsClassifier(n_neighbors=11, leaf_size=50, weights='uniform', metric='manhattan')

        # Fit + evaluation 
        models = [clf_SVM, clf_LR, clf_RF, clf_KNN]
        
        for i, classif in enumerate(classifiers):
            models[i].fit(X_UVFS, y_train_fold)  # fit models
            
            df_evaluation.loc[f'{classif} (validation) for {var_num} variables, fold {fold}',:] = eval_Performance(y_val_fold, X_UVFS_val, models[i], clf_name = classif)
            df_evaluation.loc[f'{classif} (train) for {var_num} variables, fold {fold}',:] = eval_Performance(y_train_fold, X_UVFS, models[i], clf_name = classif)
        fold += 1
    print(f'Iteration {var_num} of {len(X_train.columns)}')
 
    # Calculate mean and std of the metrics
    for i, classif in enumerate(classifiers):
        df_mean_performance.loc[f'{classif} (validation) with {var_num} features',:] = pd.DataFrame([df_evaluation.loc[f'{classif} (validation) for {var_num} variables, fold {f}',:] for f in range(1, 6)]).mean().values
        df_mean_performance.loc[f'{classif} (train) with {var_num} features',:] = pd.DataFrame([df_evaluation.loc[f'{classif} (train) for {var_num} variables, fold {f}',:] for f in range(1, 6)]).mean().values

        df_std_performance.loc[f'{classif} (validation) with {var_num} features',:] = pd.DataFrame([df_evaluation.loc[f'{classif} (validation) for {var_num} variables, fold {f}',:] for f in range(1, 6)]).std().values
        df_std_performance.loc[f'{classif} (train) with {var_num} features',:] = pd.DataFrame([df_evaluation.loc[f'{classif} (train) for {var_num} variables, fold {f}',:] for f in range(1, 6)]).std().values
        
# Export Dataframes -> to have backup
df_mean_performance.to_csv('../data/df_mean_performance.csv')
df_std_performance.to_csv('../data/df_std_performance.csv')
df_evaluation.to_csv('../data/df_evaluation.csv')
best_features_in_selection.to_csv('../data/best_features_in_selection.csv') 


### Visualizing ML performance as a function of the number of features ###########################################################################

count = 0
fig_eval, axes = plt.subplots(4, 2, figsize=(12,16))
eval_metrics_1 = ['accuracy', 'precision', 'recall', 'roc_auc']
eval_metrics_2 = ['tp','fp','tn','fn']
# Stores optimal number of features for each classifier
best_number_features = []

for i, classif in enumerate(classifiers):
    # Store means and std for each algorithm
    eval_val = df_mean_performance.iloc[count::8]
    eval_val_std = abs(df_std_performance.iloc[count::8])
    count += 1
    eval_train = df_mean_performance.iloc[count::8]
    eval_train_std = abs(df_std_performance.iloc[count::8])
    count += 1
    
    eval_val = eval_val.set_index([variables_numbers])  # necessary to plot xaxis correctly
    eval_train = eval_train.set_index([variables_numbers])
    
    # find max accuracy and number of features -> hard because no clear maximum
    max_evalu = max(np.sum(eval_val[eval_metrics_1], axis=1))  # take number of features with max sum of all eval_metrics_1
    best_numb = eval_val.loc[np.sum(eval_val[eval_metrics_1], axis=1)==max_evalu].index.to_list()
    best_number_features.append(best_numb[0])
    
    sns.lineplot(eval_val[eval_metrics_1], ax=axes[i,0])
    sns.lineplot(eval_train[eval_metrics_1], ax=axes[i,1])
    
    axes[i,0].set_title(f'{classif} validation scores')
    axes[i,1].set_title(f'{classif} training scores')

    for j in range(0,2):
        axes[i,j].set_xlabel('Number of features')
        axes[i,j].set_xticks(np.arange(0,341,25))      
        axes[i,j].set_ylabel('Score')
        axes[i,j].set_yticks(np.arange(0,1.1,0.1))
        axes[i,j].set_ylim(0, 1.1)
        axes[i,j].grid(linestyle='--')
'''  
    # add std
    for metric in eval_metrics_1:
        axes[i,0].fill_between(variables_numbers, eval_val[metric].values - 2*eval_val_std[metric].values, eval_val[metric].values + 2*eval_val_std[metric].values, alpha=0.2)  
        axes[i,1].fill_between(variables_numbers, eval_train[metric].values - 2*eval_train_std[metric].values, eval_train[metric].values + 2*eval_train_std[metric].values, alpha=0.2)  

  '''      

for k, optimal_numb in enumerate(best_number_features):
    axes[k,0].axvline(best_number_features[k], color='red')  
    print(f'Best numer of features for {classifiers[k]}: {best_number_features[k]}')

plt.tight_layout()
plt.savefig('../output/optimal_feat_num.png')       