### Import packages ################################################################################################

import pandas as pd
import numpy as np
import dataframe_image as dfi   # to export df as a table

# Visualization libraries 
import matplotlib.pyplot as plt
from matplotlib import colormaps
import seaborn as sns
import missingno as msno

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


### Import data sets ################################################################################################

df_clin = pd.read_csv('../data/clinFeatures_UPENN.csv')
df_clin = df_clin.drop(["SubjectID", "Time_since_baseline_preop"], axis=1)   # ID is redundant with index + feature is leftover from original data set (all values are 0)
df_rad = pd.read_csv('../data/radFeatures_UPENN.csv')
df_rad = df_rad.drop(["SubjectID"], axis=1)
df_key = pd.read_csv('../data/UPENN-GBM_CaPTk_fe_params.csv')


### First data inspection ################################################################################################

all_clin_features = ", ".join(df_clin.columns.values)  # all clinical features
five_rad_features = ", ".join(df_rad.columns.values[0:5])  # first 5 radiomic features

# How many columns and rows are there in the data?
print(f"The raw dataset contains {df_clin.shape[1]} clinical and {df_rad.shape[1]} radiomic features of {df_clin.shape[0]} patients.")
print(f"The clinical features are {all_clin_features}.")
print(f"Some radiomic features are {five_rad_features}.")

# Renaming 
df_clin = df_clin.replace('Not Available', np.nan)
df_clin = df_clin.replace('Unknown', np.nan)
df_clin = df_clin.replace('Indeterminate', np.nan)
df_clin = df_clin.replace('NOS/NEC', np.nan)

# Duplicates
rad_dupl = df_rad[df_rad.duplicated() == True]
clin_dupl = df_clin[df_clin.duplicated() == True]
print(f"The clinical data contains {clin_dupl.shape[0]} duplicates and the radiomic data contains {rad_dupl.shape[0]} duplicates. Furthermore, the two data sets share the same dimensions.")


### Missing values and data types ####################################################################################

# for dataframe df_clin    
data_type = df_clin.dtypes  
missing_values = df_clin.isna().sum()  
df_info = pd.DataFrame({"missing_values": missing_values, "original_data_type": data_type}) # Create an auxiliary dataframe with the values
df_info.index.name = "variable_name"
df_info["modified_data_type"] = ['category', 'float64', 'int64', 'dropped', 'dropped', 'dropped', 'category', 'dropped'] 

# missing values for dataframe df_rad
missing = df_rad.isna().sum()
missing = missing.sort_values(ascending=False)
info_radio = {}
copy = {}
pd.set_option('display.max_colwidth', None)

for i in missing.unique():
    variables = missing.loc[missing == i].index.values[:]
    copy[i] = variables
    variables = variables[:4].astype(str)
    info_radio[i] = "\n".join(variables)

df_info_radio = pd.DataFrame(info_radio.items(), columns = ['Number of missing values (per variable)', 'Variables (max. 4 are displayed)'])
df_copy = pd.DataFrame(copy.items(), columns = ['Number of missing values (per variable)', 'Variables (max. 4 are displayed)'])

df_info_radio['Number of variables'] = 0

for j in range(0, len(df_copy['Variables (max. 4 are displayed)'])):
    df_info_radio['Number of variables'][j] = len(df_copy['Variables (max. 4 are displayed)'][j])

# Dtypes for dataframe df_rad
rad_dtypes = df_rad.dtypes.unique()
var_type = [np.nan, np.nan]
var_type[0] = df_rad.dtypes.loc[df_rad.dtypes == df_rad.dtypes.unique()[0]].index
var_type[1] = df_rad.dtypes.loc[df_rad.dtypes == df_rad.dtypes.unique()[1]].index
var_num = [len(var_type[0]), len(var_type[1])]

df_rad_dtypes = pd.DataFrame({"original_data_type": rad_dtypes, "Variables (max. 4 are displayed)": var_type, 'Number of variables': var_num}) # Create an auxiliary dataframe with the values
for k in range(len(rad_dtypes)):
    four_var = df_rad_dtypes["Variables (max. 4 are displayed)"][k][:4].astype(str)
    df_rad_dtypes["Variables (max. 4 are displayed)"][k] = "\n".join(four_var)
    
# Identify individuals with most missing rows
missing_row = df_rad.isna().sum(axis=1)
missing_row = missing_row.sort_values(ascending=False)

info_radio_row = {}
copy_row = {}

for i in missing_row.unique():
    indices = missing_row.loc[missing_row == i].index.values[:]
    copy_row[i] = indices
    indices = np.sort(indices)
    indices = indices[:6].astype(str)
    info_radio_row[i] = ", ".join(indices)

df_info_radio_row = pd.DataFrame(info_radio_row.items(), columns = ['Number of missing values (per sample)', 'Samples (max. 6 sample IDs are displayed)'])
df_copy_row = pd.DataFrame(copy_row.items(), columns = ['Number of missing values (per sample)', 'Samples (max. 6 sample IDs are displayed)'])

df_info_radio_row['Number of samples'] = 0

for j in range(0, len(df_copy_row['Samples (max. 6 sample IDs are displayed)'])):
    df_info_radio_row['Number of samples'][j] = len(df_copy_row['Samples (max. 6 sample IDs are displayed)'][j])
    
# Drop missing values
df_full = pd.concat([df_clin, df_rad], axis=1)

df_full = df_full.dropna(axis=0, subset="Survival_from_surgery_days")  # drop all 159 samples with no outcome variable (452 remain)

"""
# Check again with:
missing = df_full.isna().sum()
print(missing.loc[missing != 0])
"""

df_full = df_full.drop([     
    "PsP_TP_score",
    "KPS",
    "MGMT",
    "IDH1"                   # we drop mutation status bc data is often not available (biopsy of tumor tissue + DNA analysis needed)
], axis=1)

df_full = df_full.dropna(axis=0, subset="GTR_over90percent")  # We want to keep GTR variable -> drops 28 rows (424 remain) (10 rows of them have already been dropped bc of no outcome)

# Find optimal amount of features to drop
threshold_list = []
remaining_patients = [0]
num_of_features = []

for i in range(df_full.shape[0]):    
    threshold = df_full.shape[0] - i  # drop all columns with i or more NA values
    df_full_copy = df_full.copy()
    df_full_copy = df_full_copy.dropna(axis=1, thresh=threshold)
    row_missing = df_full_copy.isna().sum(axis=1)
    row_missing = row_missing[df_full_copy.isna().sum(axis=1) != 0]    # all these patients need to be dropped subsequently
    dropped_patients = len(row_missing)
    remaining_patients.append(df_full_copy.shape[0]-dropped_patients)
    num_of_features.append(df_full_copy.shape[1])
    threshold_list.append(threshold)
    
    if remaining_patients[i] != remaining_patients[i+1]:
        print(f"Threshold: {threshold}, Features: {num_of_features[i]}, Patients: {remaining_patients[i+1]}")

remaining_patients.remove(0)  

# Visualizing NA dropping strategy
fig_missing, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8))

fig_thres1 = sns.lineplot(x=threshold_list, y=num_of_features, ax=ax1, color='black')  
fig_thres1.set(xticks= np.arange(0, 425, 50), yticks= np.arange(0, 5000, 500), title='Dropped features with varying threshold', xlabel='Threshold (required number of non-NA values per column)', ylabel='Number of features')  
ax1.axvspan(xmin=366, xmax=417, color='orange', alpha=0.2)
ax1.grid(linestyle='--')
ax1.vlines(x=0, ymin=4756, ymax=4760, color='red', label='Initial dropping of 4 features \n ("PsP_TP_score", "KPS", "MGMT", and "IDH1")')
ax1.vlines(x=424, ymin=0, ymax=0)      # needed so that yaxis still starts at 0
ax1.legend()

fig_thres2 = sns.lineplot(x=threshold_list, y=remaining_patients, ax=ax2, color='black')  
fig_thres2.set(xticks= np.arange(0, 425, 50), yticks= np.arange(0, 625, 50), title='Subsequent dropping of all patients that still have NA values', xlabel='Threshold (required number of non-NA values per column)', ylabel='Number of patients')  
ax2.axvspan(xmin=366, xmax=417, color='orange', alpha=0.2)
ax2.grid(linestyle='--')
ax2.vlines(x=424, ymin=424, ymax=611, color='red', label='Initial dropping of all patients with missing values for \n"Survival_from_surgery_days" or "GTR_over90percent" \n (187 of 611 patients)')
ax2.vlines(x=424, ymin=424, ymax=611, color='red')
ax2.vlines(x=424, ymin=0, ymax=0)      # needed so that yaxis still starts at 0
ax2.legend()

fig_missing.tight_layout()
plt.savefig('../output/fig_missing_values.png')

# Remove last missing values
df_full = df_full.dropna(axis=1, thresh=417)    # we keep all columns with 417 or more specified values (we drop all columns with 7 (=424-417) or more NA values)
df_full = df_full.dropna(axis=0)                # we drop all remaining patients with any NA values (8 patients)

print(f"The dataset with no missing values contains {df_full.shape[1]} clinical and radiomic features of {df_full.shape[0]} patients.")

# Visualize missing values of df_clin -> df_rad is to big for that :(
fig = msno.matrix(df_clin)
fig_copy = fig.get_figure()
fig_copy.savefig('../output/na_plot_df_clin.png', bbox_inches = 'tight')

### Change data types ####################################################################################

df_full = df_full.astype({
    "Gender": 'category',
    "Survival_from_surgery_days": int, 
    "GTR_over90percent": 'category',
    'Age_at_scan_years': int
})

### Data labeling ####################################################################################

median_OS_days = df_full["Survival_from_surgery_days"].median()  # we take median as threshold for calssification -> perfectly balanced classes
print(f"The median survival after surgery is {median_OS_days}.")
print("Therefore, we take 1 year (365 days) as threshold for a binary classification. Doing so, we get a well balanced label.")

df_full = df_full.reset_index(drop=True)
df_full['OS_>_1_year'] = 0

# Define categorical outcome (overall survival = OS)
for index, item in enumerate(df_full["Survival_from_surgery_days"]):
    if item > 365:     
        df_full['OS_>_1_year'][index] = 1

# Adjust categorical features (binary)
df_full["Gender"] = df_full["Gender"].cat.rename_categories({"M": 1, "F": 0})      # Male is 1 here (no front to all feminists reading this; it's just easier to remember for me)
df_full["GTR_over90percent"] = df_full["GTR_over90percent"].cat.rename_categories({"Y": 1, "N": 0}) 

# Define features and label
X = df_full.drop(['OS_>_1_year', "Survival_from_surgery_days"], axis=1)
y = df_full['OS_>_1_year']

# Export dataframes to tables (takes some time)
dfi.export(df_info,"../output/info_df_clin.png")  # add dpi = ... for higher resolution output
dfi.export(df_info_radio,"../output/info_df_rad.png")  
dfi.export(df_info_radio_row,"../output/info_df_rad_row.png")  
dfi.export(df_rad_dtypes,"../output/info_df_rad_types.png") 

### Export data for following files #############################################################
df_full.to_csv('../data/df_full.csv', index=False)
X.to_csv('../data/X.csv', index=False)
y.to_csv('../data/y.csv', index=False)