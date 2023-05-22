# # Project: Identify Customer Segments

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# magic word for producing visualizations in notebook
get_ipython().run_line_magic('matplotlib', 'inline')

'''
Import note: The classroom currently uses sklearn version 0.19.
If you need to use an imputer, it is available in sklearn.preprocessing.Imputer,
instead of sklearn.impute as in newer versions of sklearn.
'''


# ### Step 0: Load the Data

# In[2]:


azdias = pd.read_csv('Udacity_AZDIAS_Subset.csv', delimiter=';')

feat_info = pd.read_csv('AZDIAS_Feature_Summary.csv', delimiter=';')


# In[3]:


azdias.describe()


# In[4]:


# azdias.shape
feat_info


# > **Tip**: Add additional cells to keep everything in reasonably-sized chunks! Keyboard shortcut `esc --> a` (press escape to enter command mode, then press the 'A' key) adds a new cell before the active cell, and `esc --> b` adds a new cell after the active cell. If you need to convert an active cell to a markdown cell, use `esc --> m` and to convert to a code cell, use `esc --> y`. 
# 
# ## Step 1: Preprocessing
# 
# ### Step 1.1: Assess Missing Data

# In[5]:


# Identify missing or unknown data values and convert them to NaNs.
feat_info['missing_or_unknown_list'] = feat_info['missing_or_unknown'].apply(lambda x: x[1:-1].split(','))

feat_info.T


# In[6]:


for attribute,missing_values_list in zip(feat_info["attribute"], feat_info["missing_or_unknown_list"]):
    if missing_values_list[0] != "": # if the list not empty 
        for missing_value in missing_values_list:
            if missing_value.isnumeric() or missing_value.lstrip('-').isnumeric():
                missing_value = int(missing_value)
            azdias.loc[azdias[attribute] == missing_value, attribute] = np.nan


# In[7]:


# azdias


# #### Step 1.1.2: Assess Missing Data in Each Column
# In[8]:



missing_data = azdias.isnull().sum()
missing_data.sort_values(ascending=True, inplace=True)


# In[9]:


missing_data = missing_data/(azdias.shape[0])*100


# In[10]:


missing_data


# In[11]:


# Plot a histogram of missing data
plt.hist(missing_data)

plt.xlabel('Percentage of missing values')
plt.ylabel('Counts')
plt.title('Histogram of missing values')
plt.show()


# In[12]:


# Investigate patterns in the amount of missing data in each column.

missing_data.plot.barh(figsize=(15,30))
plt.xlabel('Column name with missing values')
plt.ylabel('Number of missing values')

plt.show()


# In[13]:


# Remove the outlier columns from the dataset. (You'll perform other data
# engineering tasks such as re-encoding and imputation later.)
outlier_columns = missing_data[missing_data>20].index
azdias.drop(columns=outlier_columns,axis="columns",inplace=True)


# In[14]:


print(outlier_columns )


# #### Discussion 1.1.2: Assess Missing Data in Each Column

# In[15]:


# How much data is missing in each row of the dataset?
missing_row_data = azdias.isnull().sum(axis=1)

plt.hist(missing_row_data, bins=50)

plt.xlabel('Number of missing row data values')
plt.ylabel('Counts')
plt.title('Missing data counts in rows')
plt.grid(True)
plt.show()


# In[16]:


# Write code to divide the data into two subsets based on the number of missing
# values in each row.
missing_less_30 = azdias[missing_row_data < 30].reset_index(drop=True)
missing_above_30 = azdias[missing_row_data >= 30].reset_index(drop=True)


# In[17]:
# Compare the distribution of values for at least five columns where there are
# no or few missing values, between the two subsets.
top_few_missing_cols=missing_data[missing_data < 30].index[:5]


# In[18]:
top_few_missing_cols


# In[19]:
def create_plot(column):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figwidth(15)
    ax1.set_title('missing values less than 30')
    sns.countplot(x=azdias.loc[missing_less_30.index,column],ax=ax1)

    ax2.set_title('missing values above 30')
    sns.countplot(x=azdias.loc[missing_above_30.index,column],ax=ax2)
    
    fig.suptitle(column)
    plt.show()

for i in range(top_few_missing_cols.size):
    create_plot(top_few_missing_cols[i])


# In[20]:
# Dropping rows with high missing value
azdias_high_missing = azdias.iloc[missing_above_30.index]

print(f'Total rows in azdias dataset is {azdias.shape[0]}')

# dropping rrows with high missing values
azdias = azdias[~azdias.index.isin(missing_above_30.index)]
azdias.head()

print(f'{len(azdias_high_missing)} rows greater than 30% in missing row values were dropped')
print(f'{azdias.shape[0]} rows are remaining')


# #### Discussion 1.1.3: Assess Missing Data in Each Row

# ### Step 1.2: Select and Re-Encode Features

# In[21]:
# How many features are there of each data type?
feat_info['type'].value_counts()


# #### Step 1.2.1: Re-Encode Categorical Features

# In[22]:
categorical_features = feat_info[feat_info.type == 'categorical'].attribute

categorical_features = categorical_features[ ~categorical_features.isin(outlier_columns)]

print(categorical_features)


# In[23]:
binary_categories_columns=[]
multi_level_columns=[]

for col in categorical_features:
    if azdias[col].nunique()==2:
        binary_categories_columns.append(col)
    else:
        multi_level_columns.append(col)

# In[24]:
for column in binary_categories_columns:
    print(azdias[column].value_counts())

# In[25]:
# Re-encode categorical variable(s) to be kept in the analysis.
azdias['VERS_TYP'].replace([2.0, 1.0], [1, 0], inplace=True)
azdias['ANREDE_KZ'].replace([2, 1], [1, 0], inplace=True)
azdias['OST_WEST_KZ'].replace(['W', 'O'], [1, 0], inplace=True)

# In[26]:
azdias=pd.get_dummies(data=azdias,columns=multi_level_columns)

# #### Discussion 1.2.1: Re-Encode Categorical Features
# In[27]:
# Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.
azdias['PRAEGENDE_JUGENDJAHRE'].value_counts()

# In[28]:
# Decade Mapping
decade_dic={1:1,2:1,3:2,4:2,5:3,6:3,7:3,8:4,9:4,10:5,11:5,12:5,13:5,14:6,15:6}
azdias["DECADE"]=azdias['PRAEGENDE_JUGENDJAHRE']
azdias["DECADE"].replace(decade_dic,inplace=True)

# In[29]:
azdias["DECADE"].value_counts()

# In[30]:
movement_dic={1:1,2:0,3:1,4:0,5:1,6:0,7:0,8:1,9:0,10:1,11:0,12:1,13:0,14:1,15:0}
azdias["MOVEMENT"]=azdias['PRAEGENDE_JUGENDJAHRE']
azdias["MOVEMENT"].replace(movement_dic,inplace=True)
azdias["MOVEMENT"].value_counts()


# In[31]:
azdias.drop(columns=['PRAEGENDE_JUGENDJAHRE'],axis="columns",inplace=True)

# In[32]:
# Investigate "CAMEO_INTL_2015" and engineer two new variables.
azdias['CAMEO_INTL_2015'].value_counts()

# In[33]:
def transform_wealth(x):
    return x // 10

# Life stage feature
def transform_life_stage(x):
    return x % 10


# In[34]:
azdias['WEALTH'] = pd.to_numeric(azdias['CAMEO_INTL_2015'])
azdias['LIFE_STAGE'] = pd.to_numeric(azdias['CAMEO_INTL_2015'])

azdias['WEALTH'] = azdias['WEALTH'].apply(transform_wealth)
azdias['LIFE_STAGE'] = azdias['LIFE_STAGE'].apply(transform_life_stage)


# In[35]:
# Validate Wealth and Life_Stage
azdias['WEALTH'].value_counts()
azdias['LIFE_STAGE'].value_counts()


# In[36]:
#delete the original column
azdias.drop(['CAMEO_INTL_2015'],axis=1,inplace=True)


# In[37]:
azdias.head()


# In[38]:
mixed_features = feat_info[feat_info.type == 'mixed'].attribute
mixed_features = mixed_features[ ~mixed_features.isin(['PRAEGENDE_JUGENDJAHRE', 'CAMEO_INTL_2015'])]
mixed_features = mixed_features[ ~mixed_features.isin(outlier_columns)]
mixed_features


# #### Discussion 1.2.2: Engineer Mixed-Type Features

# In[39]:
def correlated_columns_to_drop(df, min_corr_level=0.95):
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find index of feature columns with correlation greater than min_corr_level
    to_drop = [column for column in upper.columns if any(upper[column] > min_corr_level)]

    return to_drop


# In[40]:
# Do whatever you need to in order to ensure that the dataframe only contains
# the columns that should be passed to the algorithm functions.
columns_to_drop = correlated_columns_to_drop(azdias, 0.95)

columns_to_drop

# In[41]:


for column_todrop in columns_to_drop:
    if column_todrop in mixed_features.values:
        azdias.drop(column_todrop, axis=1, inplace=True)


# In[42]:
np.unique(azdias.dtypes.values)
azdias.head()


# ### Step 1.3: Create a Cleaning Function

# In[43]:
def clean_data(df, features):
    
    """
    Perform feature trimming, re-encoding, and engineering for demographics
    data
    
    INPUT: Demographics DataFrame
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """
    
    # Put in code here to execute all main cleaning steps:
    # convert missing value codes into NaNs, ...
    features['missing_or_unknown_list'] = features['missing_or_unknown'].apply(lambda x: x[1:-1].split(','))

    for attribute,missing_values_list in zip(features["attribute"], features["missing_or_unknown_list"]):
        if missing_values_list[0] != "": # if the list not empty 
            for missing_value in missing_values_list:
                if missing_value.isnumeric() or missing_value.lstrip('-').isnumeric():
                    missing_value = int(missing_value)

                df.loc[df[attribute] == missing_value, attribute] = np.nan
    # remove selected columns and rows, ...

    outlier_columns = ['ALTER_HH', 'GEBURTSJAHR', 'KBA05_BAUMAX', 'KK_KUNDENTYP', 'AGER_TYP',
       'TITEL_KZ']

    df.drop(columns=outlier_columns,axis="columns",inplace=True)
    #remove selected rows
    missing_row_data = df.isnull().sum(axis=1)
    missing_row_data.sort_values(ascending=True, inplace=True)
    missing_above_30 = df[missing_row_data >= 30].reset_index(drop=True)


    azdias_many_missing = df.iloc[missing_above_30.index]
    df = df[~df.index.isin(missing_above_30.index)]

    print(f'{len(azdias_many_missing)} rows greater than 30% in missing row values were dropped')
    print(f'{df.shape[0]} rows are remaining')

    # select, re-encode, and engineer column values.
    categorical_features = features[features.type == 'categorical'].attribute
    categorical_features = categorical_features[ ~categorical_features.isin(outlier_columns)]

    binary_categories_columns=[]
    multi_level_columns=[]

    for col in categorical_features:
        if df[col].nunique()==2:
            binary_categories_columns.append(col)
        else:
            multi_level_columns.append(col)

    df['VERS_TYP'].replace([2.0, 1.0], [1, 0], inplace=True)
    df['ANREDE_KZ'].replace([2, 1], [1, 0], inplace=True)
    df['OST_WEST_KZ'].replace(['W', 'O'], [1, 0], inplace=True)

    df=pd.get_dummies(data=df,columns=multi_level_columns)

    decade_dic={1:1,2:1,3:2,4:2,5:3,6:3,7:3,8:4,9:4,10:5,11:5,12:5,13:5,14:6,15:6}
    df["DECADE"]= df['PRAEGENDE_JUGENDJAHRE']
    df["DECADE"].replace(decade_dic,inplace=True)

    movement_dic={1:1,2:0,3:1,4:0,5:1,6:0,7:0,8:1,9:0,10:1,11:0,12:1,13:0,14:1,15:0}
    df["MOVEMENT"]= df['PRAEGENDE_JUGENDJAHRE']
    df["MOVEMENT"].replace(movement_dic,inplace=True)

    df.drop(columns=['PRAEGENDE_JUGENDJAHRE'],axis="columns",inplace=True)
    df['WEALTH'] = pd.to_numeric(df['CAMEO_INTL_2015'])
    df['LIFE_STAGE'] = pd.to_numeric(df['CAMEO_INTL_2015'])

    df['WEALTH'] = df['WEALTH'].apply(transform_wealth)
    df['LIFE_STAGE'] = df['LIFE_STAGE'].apply(transform_life_stage)

    df.drop(['CAMEO_INTL_2015'],axis=1,inplace=True)

    mixed_features = features[features.type == 'mixed'].attribute

    mixed_features = mixed_features[ ~mixed_features.isin(['PRAEGENDE_JUGENDJAHRE', 'CAMEO_INTL_2015'])]

    mixed_features = mixed_features[ ~mixed_features.isin(outlier_columns)]

    columns_to_drop = ['LP_LEBENSPHASE_GROB',
                        'LP_FAMILIE_GROB_1.0',
                        'LP_FAMILIE_GROB_2.0',
                        'LP_STATUS_GROB_5.0',
                        'MOVEMENT']

    for column_todrop in columns_to_drop:
        if column_todrop in mixed_features.values:
            df.drop(column_todrop, axis=1, inplace=True)

    if 'GEBAEUDETYP_5.0' in df.columns:
        df = df.drop(['GEBAEUDETYP_5.0'], axis=1)

    # Return the cleaned dataframe.
    return df     


# In[44]:
azdias_test = pd.read_csv('Udacity_AZDIAS_Subset.csv', delimiter=';')

feat_info_test = pd.read_csv('AZDIAS_Feature_Summary.csv', delimiter=';')
df_result = clean_data(azdias_test, feat_info_test)

# In[45]:
if 'GEBAEUDETYP_5.0' in azdias.columns:
        azdias = azdias.drop(['GEBAEUDETYP_5.0'], axis=1)

print(azdias.equals(df_result))

# ## Step 2: Feature Transformation
# In[46]:
azdias_columns = azdias.columns

# In[47]:
# If you've not yet cleaned the dataset of all NaN values, then investigate and
# do that now.
imputer = Imputer(missing_values=np.nan, strategy='mean')
azdias_imputer = imputer.fit_transform(azdias)
azdias_imputer = pd.DataFrame(azdias_imputer, columns=azdias_columns)

azdias_imputer.isnull().sum().sum()


# In[48]:
# Apply feature scaling to the general population demographics data.
scaler = StandardScaler()
azdias_scaled = scaler.fit_transform(azdias_imputer)
azdias_scaled = pd.DataFrame(azdias_scaled, columns=azdias_columns)
azdias_scaled.head()


# ### Discussion 2.1: Apply Feature Scaling

# In[49]:


# Apply PCA to the data.
pca = PCA()
pca.fit(azdias_scaled)


# In[50]:
def scree_plot(pca):
    '''
    Creates a scree plot associated with the principal components 
    
    INPUT: pca - the result of instantian of PCA in scikit learn
            
    OUTPUT:
            None
    '''
    num_components=len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
 
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)
 
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')


# In[51]:


# Investigate the variance accounted for by each principal component.
scree_plot(pca)


# In[52]:


# Re-apply PCA to the data while selecting for number of components to retain.
X = StandardScaler().fit_transform(azdias_scaled)
pca = PCA(n_components=100)
X_pca = pca.fit_transform(X)


# In[53]:
for i in np.arange(10, 101, 10):
    print(
        f'{i} components explain {pca.explained_variance_ratio_[:i].sum()} of variance.'
    )


# In[54]:
scree_plot(pca)

# ### Discussion 2.2: Perform Dimensionality Reduction

# In[55]:
# Map weights for the first principal component to corresponding feature names
# and then print the linked values, sorted by weight.
# HINT: Try defining a function here or in a new cell that you can reuse in the
# other cells.
def get_weights(pca, feature_columns, component_number):
    component = pd.DataFrame(pca.components_, columns=list(feature_columns)).iloc[component_number-1]
    component.sort_values(ascending=False, inplace=True)
    component = pd.concat([component.head(5), component.tail(5)])
    component.plot(
        kind='bar',
        title=f'Most {5 * 2} weighted features for PCA component {component_number}',
        figsize=(12, 6),
    )
    plt.show()
    return component


# In[56]:
# Map weights for the second principal component to corresponding feature names
# and then print the linked values, sorted by weight.

positive_values = []
negative_values = []

weights_1 = get_weights(pca, azdias_scaled.columns, 1)
weights_1


# In[57]:
# Map weights for the third principal component to corresponding feature names
# and then print the linked values, sorted by weight.
weights_2 = get_weights(pca, azdias_scaled.columns, 2)
weights_2


# In[58]:


weights_3 = get_weights(pca, azdias_scaled.columns, 3)
weights_3


# In[59]:
positive_values = weights_1.head().keys().tolist() + weights_2.head().keys().tolist() + weights_3.head().keys().tolist()
negative_values = weights_1.tail().keys().tolist() + weights_2.tail().keys().tolist() + weights_3.tail().keys().tolist()

print(positive_values)
print(negative_values)

# ### Discussion 2.3: Interpret Principal Components

# In[ ]:
# Over a number of different cluster counts...

centers = list(range(2, 30, 2)) 
print(centers) 
scores = []
for center in centers:

    # run k-means clustering on the data and...
    kmeans = KMeans(n_clusters=center)
    model = kmeans.fit(X_pca)
    
    # compute the average within-cluster distances.    
    # Obtain a score related to the model fitpd.read_csv('Udacity_CUSTOMERS_Subset.csv', delimiter=';')
    score = np.abs(model.score(X_pca))
    scores.append(score)  

# In[ ]:
# Investigate the change in within-cluster distance across number of clusters.
# HINT: Use matplotlib's plot function to visualize this relationship.
plt.figure(figsize=(18, 8))
plt.xticks(np.arange(0, centers[-1]+1, step=1))
plt.xlabel('K')
plt.ylabel('SSE')
plt.title('SSE vs. K')
plt.plot(centers, scores, linestyle='--', marker='o');


# In[ ]:
# Re-fit the k-means model with the selected number of clusters and obtain
# cluster predictions for the general population demographics data.
kmeans = KMeans(n_clusters=26)
model = kmeans.fit(X_pca)
population_prediction = model.predict(X_pca)


# In[ ]:
def create_kmeans_plot(data, labels): 
    fig = plt.figure(figsize=(28, 12));
    ax = Axes3D(fig)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='tab10');


# In[ ]:
create_kmeans_plot(X_pca, population_prediction)

# ### Discussion 3.1: Apply Clustering to General Population

# In[ ]:
# Load in the customer demographics data.
customers = pd.read_csv('Udacity_CUSTOMERS_Subset.csv',delimiter=';')
customers

# In[ ]:
customer_data = clean_data(customers, feat_info_test)

# In[ ]:
customer_columns = customer_data.columns


# In[ ]:
customer_imputer = imputer.transform(customer_data)
customer_imputer = pd.DataFrame(customer_imputer, columns=customer_columns)


# In[ ]:
customer_imputer.isnull().sum().sum()
customer_imputer

# In[ ]:
customer_scaled = scaler.transform(customer_imputer)
customer_scaled = pd.DataFrame(customer_scaled, columns=customer_columns)
customer_scaled.head()

# In[ ]:
customers_pca = pca.transform(customer_scaled)

# In[ ]:
#cluster predictions
customer_prediction= model.predict(customers_pca)

# In[ ]:

create_kmeans_plot(customers_pca, customer_prediction)

# ### Step 3.3: Compare Customer Data to Demographics Data
# In[ ]:
# Compare the proportion of data in each cluster for the customer data to the
# proportion of data in each cluster for the general population.
clusterno_predicted_general = pd.Series(population_prediction).value_counts().sort_index()

clusterno_predicted_customers = pd.Series(customer_prediction).value_counts().sort_index()

df_both = pd.concat([clusterno_predicted_general, clusterno_predicted_customers], axis=1).reset_index()
df_both.columns = ['clusters', 'pred_general', 'pred_customers']

df_both

# In[ ]:

df_both['general_prop'] = (df_both['pred_general']/df_both['pred_general'].sum()*100).round(3)
df_both['customers_prop'] = (df_both['pred_customers']/df_both['pred_customers'].sum()*100).round(3)

# calculating the diferences between the two proportions
df_both['difference'] = df_both['general_prop'] - df_both['customers_prop']
df_both


# In[ ]:
fig = plt.figure(figsize=(20,8))

ax = fig.add_subplot(1,1,1)

ax = df_both['general_prop'].plot(x=df_both['clusters'], kind='bar',color='orange',width=-0.3, align='edge',position=0)
ax = df_both['customers_prop'].plot(x=df_both['clusters'], kind='bar',color='blue',width = 0.3, align='edge',position=1)

ax.margins(x=0.5,y=0.1)
ax.set_xlabel('Clusters', fontsize=15) 
ax.set_ylabel('Proportions (%)', fontsize=15)
ax.tick_params(axis = 'x', which = 'major', labelsize = 14)
plt.xticks(rotation=360,)

plt.legend(('General population %', 'Customer population %'),fontsize=15)
plt.title('Comparing the proportion of general and customer populations(%) in each clusters',fontsize=16)

plt.subplots_adjust(bottom=0.2)
plt.suptitle("Barplot", fontsize=15)
plt.show()

# In[ ]:
overrepresented_customers = scaler.inverse_transform([pca.inverse_transform(model.cluster_centers_[3])]).round(0)
# In[ ]:

overrepresented_customers_df=pd.DataFrame({"feature":azdias_scaled.columns, "overrepresented_customers":overrepresented_customers[0]})
overrepresented_customers_df

# In[ ]:
overrepresented_customers_df[overrepresented_customers_df["feature"].isin(positive_values)]
# ### Overrepresented Customers positive values

# In[ ]:
overrepresented_customers_df[overrepresented_customers_df["feature"].isin(negative_values)]

# ### Underrepresented Customers positive values
underrepresented_customers = scaler.inverse_transform([pca.inverse_transform(model.cluster_centers_[9])]).round(0)
# In[ ]:


underrepresented_customers_df=pd.DataFrame({"feature":azdias_scaled.columns, "underrepresented_customers":underrepresented_customers[0]})
underrepresented_customers_df

# In[ ]:

underrepresented_customers_df[underrepresented_customers_df["feature"].isin(positive_values)]

# In[ ]:
underrepresented_customers_df[underrepresented_customers_df["feature"].isin(negative_values)]
