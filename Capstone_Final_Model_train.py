#!/usr/bin/env python
# coding: utf-8

### Best Model for Predicting NYC Motor Vehicle Collisions -  

## Loading all the basic libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import calendar
import pickle


## Parameters - 
# specifying the number of folds to be used:
n_splits = 5
# creating a file where we'll write it:
output_file = f'model.bin'



## Data Preparation -
# importing the data from source as a csv file and converting it into a Pandas DataFrame:
data = pd.read_csv('NYC_Motor_Vehicle_Collisions_to_Person.csv')
# viewing the snapshot of first 5 rows of the loaded dataset:
data.head()



## Data Cleaning and Formatting -
# dropping irrelevant columns from the dataset and assigning remaining dataset to a variable:
NYC_df = data.drop(['VEHICLE_ID', 'PERSON_ID','UNIQUE_ID','COLLISION_ID'], axis = 1)

# imputing missing values in PERSON_AGE column with mean PERSON_AGE values:
NYC_df['PERSON_AGE'] = NYC_df['PERSON_AGE'].fillna(np.mean(data['PERSON_AGE']))

# imputing missing values in Other columns with 'Unknown' or Most common values:
NYC_df['SAFETY_EQUIPMENT'].fillna("Unknown",inplace = True)
NYC_df['PED_LOCATION'].fillna("Unknown",inplace = True)
NYC_df['CONTRIBUTING_FACTOR_2'].fillna("Unspecified",inplace = True)
NYC_df['EJECTION'].fillna("Not Ejected",inplace = True)
NYC_df['CONTRIBUTING_FACTOR_1'].fillna("Unspecified",inplace = True)
NYC_df['POSITION_IN_VEHICLE'].fillna("Unknown",inplace = True)
NYC_df['PED_ACTION'].fillna("Unknown",inplace = True)

# converting the 'Crash Date' column to datetime format:
NYC_df['CRASH_DATE']= pd.to_datetime(NYC_df['CRASH_DATE'])

# converting the 'Person Age' column to integer format:
NYC_df['PERSON_AGE']= NYC_df['PERSON_AGE'].astype('int64')

# replacing some values in each column with specific values:
# changing "Does Not Apply" to "Unknown"or "None" 
NYC_df['BODILY_INJURY'].replace('Does Not Apply','None',inplace=True)  
NYC_df['PERSON_SEX'].replace('U','M',inplace=True)
NYC_df['PED_LOCATION'].replace('Does Not Apply','Unknown',inplace=True)
NYC_df['COMPLAINT'].replace('Does Not Apply','Unknown',inplace=True)
NYC_df['EMOTIONAL_STATUS'].replace('Does Not Apply','Unknown',inplace=True)
NYC_df['PED_ACTION'].replace('Does Not Apply','Unknown',inplace=True)
NYC_df['PERSON_INJURY'].replace({'Injured': 0 ,'Killed': 1},inplace=True)

# converting Months to abbreviated Month Names:
month_name = []
# extracting Months from Crash_Date and using it to get abbreviated month names using for loop:
crash_month = pd.DatetimeIndex(NYC_df['CRASH_DATE']).month
for i in crash_month:
    mnth_abb = calendar.month_abbr[i]
    month_name.append(mnth_abb)
# assigning month name to a column
NYC_df['CRASH_Mnth_Name'] = month_name




## Splitting the Data and getting the Feature Matrix & Target variables -
# splitting the dataset using sklearn into 60-20-20:
# Step 1 - splitting dataset into full train and test subsets first:
df_full_train, df_test = train_test_split(NYC_df, test_size=0.2,random_state=1) 

# Step 2 - splitting full train subset again into training set and validation set:
df_train, df_val = train_test_split(df_full_train, test_size=0.25,random_state = 1)

# resetting indices for each of the subset: 
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# getting our target variable column ('PERSON_INJURY') subsets as respective Numpy arrays:
y_train = df_train.PERSON_INJURY
y_val = df_val.PERSON_INJURY
y_test = df_test.PERSON_INJURY

# deleting 'PERSON_INJURY' or target column from feature matrix subsets:
del df_train['PERSON_INJURY']
del df_val['PERSON_INJURY']
del df_test['PERSON_INJURY']

# re-checking the size of 3 subsets after deleting the target column:
df_train.shape, df_val.shape, df_test.shape




## Predicting on Test data using our Final Model (Random Forest for Classification) -
# resetting indices of full_train DataFrame:
df_full_train = df_full_train.reset_index(drop=True)

# encoding string class values of target variable PERSON_INJURY column as integers - using LabelEncoder():
label_encoder = LabelEncoder()
y_full_train = label_encoder.fit_transform(df_full_train.PERSON_INJURY)

# converting the CRASH_DATE column as a Timestamp and then converting it to an integer data type for 
# df_full_train and df_test subsets:
df_full_train['CRASH_DATE'] = df_full_train['CRASH_DATE'].map(pd.Timestamp.timestamp).astype(int)
df_test['CRASH_DATE'] = df_test['CRASH_DATE'].map(pd.Timestamp.timestamp).astype(int)

# turning the full train df into dictionaries:
dicts_full_train = df_full_train.to_dict(orient='records')

# instantiating the vectorizer instance:
dv = DictVectorizer(sparse=False)

# turning list of dictionaries into full train feature matrix:
X_full_train = dv.fit_transform(dicts_full_train)

# turning the test df into dictionaries:
dicts_test = df_test.to_dict(orient='records')

# turning list of dictionaries into testing feature matrix:
X_test = dv.transform(dicts_test)

# specifying our best model Random Forest Classifier parameters as a variable, rf:
rf = RandomForestClassifier(random_state=42, max_depth=15, min_samples_leaf = 1, max_features = 8, n_estimators = 70)

# training our full_train set with above best model Random Forest Classifier parameters:
model = rf.fit(X_full_train, y_full_train)

# predicting using our best model Random Forest for Classification on the testing set:
y_pred2 = rf.predict_proba(X_test)[:,1]

# computing the AUC score on testing set:
print('AUC on test Rand_Forest: %.3f' % roc_auc_score(y_test, y_pred2))




## Training -
# Using KFold Cross-Validation on our Final Model for making Predictions - 
# Step 1 -
# Function 1 - Creating a function to train our DataFrame:
def train(df_train, y_train):
    dicts = df_train.to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    
    model = rf.fit(X_train, y_train)
    
    return dv, model

# Step 2 - 
# Function 2 - Creating another function to predict:
def predict(df, dv, model):
    dicts = df.to_dict(orient='records')  # converts df to list of dictionaries
    
    X = dv.transform(dicts)  # creates a feature matrix using the vectorizer
    y_pred = model.predict(X) # uses the model
    
    return y_pred


## Validation -
print(f'doing validation')

# Performing K-fold Cross validation and evaluating the AUC scores after each iteration:
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

scores = []
fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    
    # selecting part of dataset as 3 subsets for model:
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]
    
    # specifying our target variable values as Numpy array for train and validation sets:
    y_train = df_train.PERSON_INJURY.values
    y_val = df_val.PERSON_INJURY.values
    
    dv, model = train(df_train, y_train)   # using train function created
    y_pred = predict(df_val, dv, model)   # using predict function created
    
    # computing auc scores for each iteration or fold in KFold:
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f'auc on fold {fold} is {auc}')
    fold= fold + 1
    
# computing mean of AUC scores and spread of AUC score:
print('validation results:')
print('%.3f +- %.3f' % (np.mean(scores), np.std(scores)))




## Training the Final Model -
print('training the final model')

# Now, Training our Final Model on Full train dataset and evaluating on test dataset -
dv, model = train(df_full_train, y_full_train)   # using train function created
y_pred = predict(df_test, dv, model)   # using predict function created

# computing auc for ROC Curve:
auc = roc_auc_score(y_test, y_pred)
print(f'auc={auc}')



## Saving the Model -
# Step 1 - taking our model and writing it to a file - 
# creating a file where we'll write it:
output_file

# writing a Binary file using pickle - alternative to open and close codes we use with open to automatically open-close a file:
with open(output_file, 'wb') as f_out:    # file output
    pickle.dump((dv, model), f_out)


print(f'the model is saved to {output_file}')


