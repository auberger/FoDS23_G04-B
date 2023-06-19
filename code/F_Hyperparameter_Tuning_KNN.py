### Import packages ################################################################################################

import pandas as pd
import numpy as np

# Sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif

# Visualization libraries 
import matplotlib.pyplot as plt
from matplotlib import colormaps
import seaborn as sns

# Import previous data
df_full = pd.read_csv('../data/df_full.csv')
X_train = pd.read_csv('../data/X_train.csv')
y_train = pd.read_csv('../data/y_train.csv')
best_number_features = [81, 12, 24, 18]

import warnings
warnings.filterwarnings("ignore")


### Manual hyperparameter tuning for KNN (elbow plot) #################################################################################

# Store numerical columns (of features)
num_columns = X_train._get_numeric_data().columns

mean_error_rate = []    # empty array for storing error rates
k_numbers = [x for x in range(1,61)]

# finding optimal k in KNN by cross validation with elbow method
kf = KFold(n_splits=5, shuffle=True, random_state=10)   # no stratified CV needed since we have perfectly balanced classes

for i in k_numbers:
    error_rate = []
    
    for train_index, val_index in kf.split(X_train): # We do the 5-fold CV on the training set only, since the hyperparameter tuning should leave out the test data!

        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
    
        # Standardize the numerical features 
        sc = StandardScaler()
        X_train_fold[num_columns] = sc.fit_transform(X_train_fold[num_columns])
        X_val_fold[num_columns]  = sc.transform(X_val_fold[num_columns])

        # Select best features
        UVFS_Selector = SelectKBest(f_classif, k=best_number_features[3])                
        X_train_KNN = UVFS_Selector.fit_transform(X_train_fold, y_train_fold)      
        X_val_KNN = UVFS_Selector.transform(X_val_fold)

        # fit the model
        clf_KNN = KNeighborsClassifier(n_neighbors=i)
        clf_KNN.fit(X_train_KNN, y_train_fold)
        pred_i = clf_KNN.predict(X_val_KNN)
        error_rate.append(np.mean(pred_i != y_val_fold.values.ravel()))
        
    mean_error_rate.append(np.mean(error_rate))
    
plt.figure(figsize=(10,6))
plt.plot(range(1,61),mean_error_rate,color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error rate vs. number of neighbors K in KNN')
plt.xlabel('Number of neighbors K')
plt.ylabel('Error rate')
plt.axvline(41, color='red')  
plt.savefig("knn_elbow_plot.png")

# Choose optimal k according to the elbow plot
optimal_k = 41
print(f'The optimal number of neighbors K seems to be {optimal_k}.')
