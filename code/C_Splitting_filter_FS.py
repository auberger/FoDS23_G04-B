### Import packages ################################################################################################

import pandas as pd
import numpy as np
import dataframe_image as dfi   # to export df as a table
import scipy.stats as sts

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import previous data
df_full = pd.read_csv('../data/df_full.csv')
X = pd.read_csv('../data/X.csv')
y = pd.read_csv('../data/y.csv')

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# Store numerical and categorical columns (of features)
num_columns = X._get_numeric_data().columns
cat_columns = list(set(X.columns)-set(num_columns))

# split data in train + test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) 

# Standardize the numerical features 
sc = StandardScaler()
X_train[num_columns] = sc.fit_transform(X_train[num_columns])
X_test[num_columns]  = sc.transform(X_test[num_columns])

### Dimensionality reduction (Filter method feature selection) #################################################################################

### Filter features by variance ################
variance = X_train.var()
variance = variance.loc[variance<0.1] # less than 0.1% variance -> feature is almost constant 
X_train = X_train.drop(variance.index, axis=1)     # drop 96 features with very low variance (can't explain much variation in outcome)
X_test = X_test.drop(variance.index, axis=1)

# Drop correlating features ################
drop_features = []
num_columns = X_train._get_numeric_data().columns  # we can only check correlation between numerical variables (excludes GTR and Gender variable)

for d, feature in enumerate(num_columns):  
    for comp_feature in num_columns[d+1:]:
        r_sprm,_  = sts.spearmanr(X_train[feature], X_train[comp_feature])
        if abs(r_sprm) > 0.8:
            drop_features.append(feature)
            break

X_train = X_train.drop(drop_features, axis=1)     # drop 1302 correlating features (reduce multi-collinearity)
X_test = X_test.drop(drop_features, axis=1)
print(f'The dataset has {X_train.shape[1]} features after the first dimensionality reduction.')

### Export data for following files #############################################################
X_train.to_csv('../data/X_train.csv', index=False)
X_test.to_csv('../data/X_test.csv', index=False)
y_train.to_csv('../data/y_train.csv', index=False)
y_test.to_csv('../data/y_test.csv', index=False)