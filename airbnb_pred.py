#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np


# In[2]:


df = pd.read_csv("listings.csv")

df.head()


# In[3]:


df.columns


# In[4]:


pd.set_option('display.max_columns', None)

print(df.dtypes)


# In[5]:


missing_values = df.isnull().sum()
print(missing_values[missing_values >0])


# In[6]:


import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Fill numerical columns with median
numerical_col = ['bathrooms', 'bedrooms', 'beds', 'reviews_per_month']
for col in numerical_col:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical columns with the most frequent value
categorical_col = ['neighbourhood', 'host_listings_count']
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_col] = cat_imputer.fit_transform(df[categorical_col])

# Replace other non-important categorical columns with 'Unknown'
other_col = ['host_name', 'host_location', 'host_about', 'host_response_time', 'host_is_superhost']
for col in other_col:
    df[col] = df[col].fillna('Unknown')

# Drop unnecessary columns that are text-heavy
drop_col = ['description', 'neighborhood_overview', 'picture_url',
            'host_thumbnail_url', 'host_picture_url', 'calendar_updated',
            'first_review', 'last_review', 'license',
            'host_name', 'host_location', 'host_about']
df.drop(columns=drop_col, inplace=True)

# Label Encoding for high-cardinality categorical columns
exclude_cols = ['price', 'listing_url', 'name']
cat_cols_remaining = [col for col in df.select_dtypes(include='object').columns if col not in exclude_cols]
#cat_cols_remaining.remove('price')  # Don't touch price column

le = LabelEncoder()
for col in cat_cols_remaining:
    df[col] = le.fit_transform(df[col])

# Now, check memory usage and dataframe info
print(df.info())


# In[7]:


print(df.dtypes)


# In[8]:


# Clean the 'price' column to remove dollar signs and commas, then convert to float
df['price'] = df['price'].replace({'\$': '', ',': ''}, regex=True).astype(float)

# Verify the transformation
print(df['price'].head())


# In[9]:


import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

df['neighbourhood']=le.fit_transform(df['neighbourhood'])

df['bedroom_bathroom_interaction'] = df['bedrooms'] * df['bathrooms']
df['bedroom_bathroom_beds_interaction'] = df['bedrooms'] * df['bathrooms'] * df['beds']
df['bed_bath_ratio'] = df['bedrooms'] / (df['bathrooms'] + 1)  # Adding 1 to avoid division by 0
df['bed_bedrooms_ratio'] = df['beds'] / (df['bedrooms'] + 1)  # Adding 1 to avoid division by 0

# Select the features you want to include for training (including interaction terms)
features = ['bathrooms', 'bedrooms', 'beds', 'reviews_per_month', 'host_listings_count', 
            'bedroom_bathroom_interaction', 'bedroom_bathroom_beds_interaction', 
            'bed_bath_ratio', 'bed_bedrooms_ratio', 'neighbourhood']

target='price'


# In[10]:


X=df[features]
y=df[target]

print(X.isnull().sum())
print(X.dtypes)


# In[11]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.isnull().sum())
print(X_val.isnull().sum())
print(X_train.dtypes)


# In[12]:


X_train = X_train.astype(float)
X_val = X_val.astype(float)
y_train = y_train.astype(float)
y_val = y_val.astype(float)


# In[13]:


print(y_train.shape)


# In[14]:


# Convert X_train and X_val to NumPy arrays
X_train_np = X_train.to_numpy()
X_val_np = X_val.to_numpy()
y_train = y_train.to_numpy()
y_val = y_val.to_numpy()



# In[15]:


print(X_train_np.shape)  # Should be (30227, n_features)
print(X_val_np.shape)  


# In[16]:


import numpy as np

# Check for NaN values in the target variable
print(np.isnan(y_train).sum())
print(np.isnan(y_val).sum())


# In[17]:


# Replace NaN values in target with the mean of the respective array
mean_y_train = np.nanmean(y_train)
mean_y_val = np.nanmean(y_val)

y_train = np.nan_to_num(y_train, nan=mean_y_train)
y_val = np.nan_to_num(y_val, nan=mean_y_val)


# In[18]:


dtrain = xgb.DMatrix(X_train_np, label=y_train)
dval = xgb.DMatrix(X_val_np, label=y_val)

# Check if the DMatrix is created successfully
print(f"DMatrix for training: {dtrain.num_row()} rows, {dtrain.num_col()} columns.")
print(f"DMatrix for validation: {dval.num_row()} rows, {dval.num_col()} columns.")


# In[19]:


import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# Initialize parameters
params = {
    'n_estimators': 1000, 
    'learning_rate': 0.05, 
    'max_depth': 6, 
    'random_state': 42,
    'eval_metric': 'mae'  # Explicitly setting eval_metric in the params
}

# Initialize the model with params
model = xgb.XGBRegressor(**params)

# Train the model with eval_set
model.fit(X_train, y_train, 
          eval_set=[(X_val, y_val)], 
          verbose=False)

# Predict on the validation set
y_pred = model.predict(X_val)

# Evaluate performance (Mean Absolute Error)
error = mean_absolute_error(y_val, y_pred)
print(f'Mean Absolute Error: {error}')




# In[20]:


X_missing = df[df['price'].isna()][X_train.columns]

#X_missing_np = X_missing.to_numpy()
predicted_prices = model.predict(X_missing)

df_new = df

df_new.loc[df_new['price'].isna(), 'price']=predicted_prices


# In[21]:


print(df['price'].isna().sum())  # Should be 0


# In[22]:


df_new.to_csv('cleaned_data_nprices.csv', index=False)
print("file saved!")


# In[ ]:




