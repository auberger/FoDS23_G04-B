### Import packages ################################################################################################

import pandas as pd
import numpy as np
import dataframe_image as dfi   # to export df as a table
import scipy.stats as sts

# Visualization libraries 
import matplotlib.pyplot as plt
from matplotlib import colormaps
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import missingno as msno
from mpl_toolkits.axes_grid1 import make_axes_locatable # used to arrange confusion matrices in subplots

# Import previous data
df_full = pd.read_csv('../data/df_full.csv')
X = pd.read_csv('../data/X.csv')

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


### Overview ####################################################################################
gender_counts = df_full["Gender"].value_counts()
gtr_count = df_full["GTR_over90percent"].value_counts()

# Store numerical and categorical columns (of features)
num_columns = X._get_numeric_data().columns
cat_columns = list(set(X.columns)-set(num_columns))

print(df_full.describe())
print(f"The final data includes {gender_counts[0]} male and {gender_counts[1]} female patients.")
print(f"The final data includes {gtr_count[0]} patients with a gross tumor resection of more than 90% and {gtr_count[1]} patients with a gross tumor resection of less than 90%.")


### Testing for normality ####################################################################################

normal = {}
nonnormal = {}
alpha = 0.05/(len(num_columns)+1)   # take multiple testing into account

for var in num_columns:
    pvalue = sts.shapiro(df_full[var]).pvalue
    if pvalue < alpha:
        nonnormal[var] = pvalue
    else:
        normal[var] = pvalue

normal_var = normal.keys()
nonnormal_var = nonnormal.keys()

print(f"Using the Shapiro-Wilk test with sign. level {alpha:.6f}, we assume that there are in total {len(normal_var)} normal and {len(nonnormal_var)-1} non-normal distributed features.")
print("The Age of patients seems to be normally distributed.")

# Distribution of the label
pvalue = sts.shapiro(df_full["Survival_from_surgery_days"]).pvalue
if pvalue < alpha:
    nonnormal["Survival_from_surgery_days"] = pvalue
    print(f"The label is not normally distributed.")
else:
    normal["Survival_from_surgery_days"] = pvalue
    print(f"The label is normally distributed.")
    
    
### Age distribution and gender ####################################################################################

age = sns.histplot(df_full, x = 'Age_at_scan_years', binwidth=5, binrange=(15,95),hue='Gender')
label_axes = age.set(xticks= np.arange(15, 100, 5), xlabel='Age [years]', ylabel='Number of Patiens', title= 'Age distribution') 
plt.legend(title='Gender', loc='upper left', labels=['Male', 'Female'])
plt.savefig("../output/age_distribution.png")


### Survival rate as a function of age and gender ####################################################################################
fig, axes = plt.subplots(1, 2, figsize=(12,6))

# Plot the age distirbution separated by outcome
fig_age = sns.histplot(df_full, x = 'Age_at_scan_years', binwidth=5, binrange=(15,95), hue='OS_>_1_year', ax=axes[0])
label_axes = fig_age.set(xticks= np.arange(15, 100, 5), xlabel='Age [years]', ylabel='Number of Patiens', title= 'Age distribution by overall survival') 
axes[0].legend(title='Overall Survival (OS)', labels=['> 1 year', '< 1 year'])

# Plot the age dispersion separated by gender
fig_age_range = sns.boxplot(data=df_full, x='Age_at_scan_years', y='Gender', hue="OS_>_1_year",  ax=axes[1], width=0.4)
fig_age_range = fig_age_range.set(xticks= np.arange(15, 105, 5), xlabel='Age [years]', ylabel=' ', title= 'Age range split by gender and overall survival')  
axes[1].set_yticklabels(['Female','Male'])
axes[1].legend(title='Overall Survival (OS)', labels=['> 1 year', '< 1 year'])

plt.savefig("../output/age_distribution_outcome.png")


### Influence of GTR (gross tumor resection) on overall survival ####################################################################################

# Importance of GTR (but this cannot really be considered for feature selection since all data is included in analysis!)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

gtr = sns.boxplot(data=df_full, x="GTR_over90percent", y="Survival_from_surgery_days", width=0.4, ax=ax1)
gtr.set(title='Overall survival (in days) among GTR-groups', xlabel='Gross Tumor resection > 90%', ylabel='Survival after surgery (in days)')
ax1.set_xticklabels(['No','Yes'])

# we use Wilcoxon bc data is skewed
test_gtr = sts.ranksums(df_full.loc[df_full["GTR_over90percent"] == 1]["Survival_from_surgery_days"], 
                        df_full.loc[df_full["GTR_over90percent"] == 0]["Survival_from_surgery_days"])
print(f"The Wilcoxon signed rank test could detect a significant difference between the mean overall survival among GTR-groups (p-value = {test_gtr[1]:0.6f}).")
print("-> Therefore, this is an important feature.")

gtr2 = sns.countplot(data=df_full, x="GTR_over90percent", hue='OS_>_1_year', ax=ax2)
gtr2.set(title='Number of patients among GTR-groups dependent on overall survival', xlabel='Gross Tumor resection > 90%', ylabel='Number of patients')
ax2.legend(title='Overall Survival (OS)', labels=['< 1 year', '> 1 year'])
ax2.set_xticklabels(['No','Yes'])

# Add percentages above each bar
yes_gtr = df_full.loc[df_full["GTR_over90percent"]==1]
no_gtr = df_full.loc[df_full["GTR_over90percent"]==0]
yes_gtr_yes_OS = yes_gtr.loc[yes_gtr['OS_>_1_year']==1]
no_gtr_no_OS = no_gtr.loc[no_gtr['OS_>_1_year']==1]

yes_gtr_yes_OS_percent = round(yes_gtr_yes_OS.shape[0] / yes_gtr.shape[0] * 100, 1)
no_gtr_no_OS = round(no_gtr_no_OS.shape[0] / no_gtr.shape[0] * 100, 1)

yes_gtr_no_OS_percent = 100 - yes_gtr_yes_OS_percent
no_gtr_yes_OS = 100 - no_gtr_no_OS

percentages = [str(no_gtr_yes_OS) + '%', str(yes_gtr_no_OS_percent) + '%', str(no_gtr_no_OS) + '%', str(yes_gtr_yes_OS_percent) + '%']

for i, p in enumerate(ax2.patches):
    height = p.get_height()
    plt.text(p.get_x() + p.get_width() / 2., height, percentages[i], ha='center', va='bottom')
    
plt.savefig('../output/gtr.png', dpi = 200)


### Cummulative deaths over time ####################################################################################

cum_deaths = np.zeros(df_full['Survival_from_surgery_days'].max()+1, int)
patients = df_full.shape[0]

for day in range(1, df_full['Survival_from_surgery_days'].max()+1):
    cum_deaths[day] = cum_deaths[day-1] + df_full.loc[round(df_full['Survival_from_surgery_days']) == day].shape[0]

cum_deaths_rel = cum_deaths / patients * 100

# Create a line plot
fig = make_subplots(rows=1, cols=1)
fig.add_trace(go.Scatter(x=np.arange(df_full['Survival_from_surgery_days'].max()+1), y=cum_deaths_rel), row=1, col=1)
fig.update_yaxes(title_text="Cumulative deaths of all patients (n=312) [%]", row=1, col=1)
fig.update_xaxes(title_text="Time after the surgery [days]", row=1, col=1)
fig.add_shape(type="line",x0=0, y0=50, x1=366, y1=50, line_width=1, line_color="red", line_dash='dash')
fig.add_shape(type="line",x0=366, y0=0, x1=366, y1=50, line_width=1, line_color="red", line_dash='dash')
fig.add_trace(go.Scatter(x=[366],y=[50], mode="markers+text", text=" After 366 days 50% of all patients are dead", textposition="bottom right"))
fig.update_layout(title= "Cumulative deaths after surgery")
fig.update_layout(xaxis = dict(tickmode = 'linear', dtick = 100), yaxis = dict(tickmode = 'linear', dtick = 10))
fig.update_layout(showlegend=False)

fig.write_image("../output/cum_deaths.png")
fig.show()