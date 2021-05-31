# Analytics Vidhya JOB-A-THON - May 2021
# 
# https://datahack.analyticsvidhya.com/contest/job-a-thon-2/

# Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sklearn.metrics as metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

sns.set(style='darkgrid')
# pd.options.display.max_rows=None
pd.options.mode.chained_assignment = None


# Read Train and Test Data

# Definition of all the variables in Train and Test Data:**
# - ID --> Unique Identifier for a row
# - Gender --> Gender of the Customer
# - Age --> Age of the Customer (in Years)
# - Region_Code --> Code of the Region for the customers
# - Occupation --> Occupation Type for the customer
# - Channel_Code --> Acquisition Channel Code for the Customer  (Encoded)
# - Vintage --> Vintage for the Customer (Account Length) (In Months)
# - Credit_Product --> If the Customer has any active credit product (Home loan, Personal loan, Credit Card etc.)
# - Avg_Account_Balance --> Average Account Balance for the Customer in last 12 Months
# - Is_Active --> If the Customer is Active in last 3 Months
# - Is_Lead(Target) --> If the Customer is interested for the Credit Card
#  - 0 : Customer is not interested
#  - 1 : Customer is interested


train_data = pd.read_csv('data/train_s3TEQDk.csv')
train_data = train_data.set_index('ID')
train_data.head()


test_data = pd.read_csv('data/test_mSzZ8RL.csv')
test_data = test_data.set_index('ID')
test_data.head()



# Identifying Numerical and Categorical columns
num_cols = ['Age', 'Vintage','Avg_Account_Balance']
cat_cols = [c for c in train_data.columns if c not in num_cols and c!='Is_Lead']
cat_cols


# Exploring Data

# Is_Lead (Target Variable)
# - Indicates if customer is interested in credit cards or not

sns.countplot(train_data['Is_Lead'])
plt.show()

train_data['Is_Lead'].value_counts(normalize=True)


# - Approximately 76-24 split in the target variable, so it is a fairly balanced dataset
# - Lets plot against each of the other variables individually to see if we can notice any patterns


fig,ax = plt.subplots(3,2,figsize=(16,15))
ax = ax.flatten()
for i,col in enumerate(cat_cols):
    sns.countplot(x=col, hue='Is_Lead', data=train_data, ax=ax[i])

fig,ax = plt.subplots(2,2,figsize=(16,10))
ax=ax.flatten()
for i,col in enumerate(num_cols):
    sns.boxplot(x='Is_Lead', y=col, data=train_data, ax=ax[i])
plt.show()

# - Customers interested in credit card (Is_Lead = 1) have on average higher age and higher vintage than those that are not interested (Is_Lead = 0)
# - So both Age and Vintage must be important variables
# - But other than that, no other variable seems to have a significant impact on the Target Variable

# Missing Values

train_data.isnull().sum()
test_data.isnull().sum()


# - Missing Values exist only in the *Credit_Product* column; This is handled later

# - First, let's look at each variables independently
# - All Categorical variables should be label encoded

# Gender
train_data['Gender'].value_counts()

# Label Encoding
gender_encoding = {'Male':0, 'Female':1}
train_data['Gender'].replace(gender_encoding, inplace=True)
test_data['Gender'].replace(gender_encoding, inplace=True)
train_data['Gender'].value_counts()


# Region_Code
train_data['Region_Code'].value_counts().sort_index()

# Label Encoding
rcode_encoding = lambda x: int(x[2:])-249
train_data['Region_Code'] = train_data['Region_Code'].apply(rcode_encoding)
test_data['Region_Code'] = test_data['Region_Code'].apply(rcode_encoding)
train_data['Region_Code'].value_counts().sort_index()


# Occupation
train_data['Occupation'].value_counts()

# Label Encoding
occ_encoding = {'Other':0, 'Self_Employed':1, 'Entrepreneur':2, 'Salaried':3}
train_data['Occupation'].replace(occ_encoding, inplace=True)
test_data['Occupation'].replace(occ_encoding, inplace=True)
train_data['Occupation'].value_counts()


# Channel_Code
train_data['Channel_Code'].value_counts()

# Label Encoding
ccode_encoding = lambda x: int(x[1:])
train_data['Channel_Code'] = train_data['Channel_Code'].apply(ccode_encoding)
test_data['Channel_Code'] = test_data['Channel_Code'].apply(ccode_encoding)
train_data['Channel_Code'].value_counts().sort_index()


# Is_Active
train_data['Is_Active'].value_counts()

# Label Encoding
isactive_encoding = {'No':0, 'Yes':1}
train_data['Is_Active'].replace(isactive_encoding, inplace=True)
test_data['Is_Active'].replace(isactive_encoding, inplace=True)
train_data['Is_Active'].value_counts()


# - For numerical variables *Age* and *Vintage*, I have tried to do binning but the results were worse so it's better to leave them as numerical variables
# - Log Transformation has improved the scores
# - Scaling the numerical variables will be done later

# Age

plt.figure(figsize=(18,6))
sns.distplot(train_data['Age'])
plt.show()

# - Applying Log Transformation to normalize data
log_transform = lambda x: np.log(x+1)
train_data['Age_log'] = train_data['Age'].apply(log_transform)
test_data['Age_log'] = test_data['Age'].apply(log_transform)

plt.figure(figsize=(18,6))
sns.distplot(train_data['Age_log'])
plt.show()


# Vintage
plt.figure(figsize=(18,6))
sns.distplot(train_data['Vintage'])
plt.show()

# - Applying Log Transformation to normalize data
log_transform = lambda x: np.log(x+1)
train_data['Vintage_log'] = train_data['Vintage'].apply(log_transform)
test_data['Vintage_log'] = test_data['Vintage'].apply(log_transform)

plt.figure(figsize=(18,6))
sns.distplot(train_data['Vintage_log'])
plt.show()

# Avg_Account_Balance
plt.figure(figsize=(18,6))
sns.distplot(train_data['Avg_Account_Balance'])
plt.show()

# - Applying Log Transformation to normalize data
log_transform = lambda x: np.log(x+1)
train_data['Avg_Account_Balance_log'] = train_data['Avg_Account_Balance'].apply(log_transform)
test_data['Avg_Account_Balance_log'] = test_data['Avg_Account_Balance'].apply(log_transform)

plt.figure(figsize=(18,6))
sns.distplot(train_data['Avg_Account_Balance_log'])
plt.show()


train_data.drop(columns=['Age','Vintage','Avg_Account_Balance'], inplace=True)
test_data.drop(columns=['Age','Vintage','Avg_Account_Balance'], inplace=True)

num_cols = [x+'_log' for x in num_cols]


# Credit_Product

train_data['Credit_Product'].fillna('Missing', inplace=True)
test_data['Credit_Product'].fillna('Missing', inplace=True)

train_data['Credit_Product'].value_counts(normalize=True)


# - Has >10% Missing values, so these rows cannot be ignored
# - Let's look at relationship with other variables independently

fig,ax = plt.subplots(3,2,figsize=(16,15))
ax = ax.flatten()
for i,col in enumerate([c for c in cat_cols if c != 'Credit_Product']):
    sns.countplot(x=col, hue='Credit_Product', data=train_data, ax=ax[i], hue_order=['No','Yes','Missing'])

fig,ax = plt.subplots(2,2,figsize=(16,10))
ax=ax.flatten()
for i,col in enumerate(num_cols):
    sns.boxplot(x='Credit_Product', y=col, data=train_data, ax=ax[i], order=['No','Yes','Missing'])
plt.show()

# - Missing values have an even higher median age and vintage; closer to those with Credit_Product=1
# - First, I tried to replace all missing values with 1, which gave good ROC AUC scores
# - KNNImputer was attempted but this made ROC AUC scores for all models worse than with previous method
# - But, converting all missing values as a new category gave much better results than any other method

# Imputing Missing Values

# Label Encoding
cp_encoding = {'No':0, 'Yes':2,'Missing':1}
train_data['Credit_Product'].replace(cp_encoding, inplace=True)
test_data['Credit_Product'].replace(cp_encoding, inplace=True)
train_data['Credit_Product'].value_counts(dropna=False, normalize=True)

# train_data['Credit_Product'] = train_data['Credit_Product'].fillna(1).astype(int)
# test_data['Credit_Product'] = test_data['Credit_Product'].fillna(1).astype(int)

# impute_cols = [x for x in train_data.columns if x not in ['Is_Lead', 'Region_Code', 'Occupation', 'Channel_Code']]
# imputer = KNNImputer()
# train_data[impute_cols] = imputer.fit_transform(train_data[impute_cols])
# test_data[impute_cols] = imputer.transform(test_data[impute_cols])
# train_data['Credit_Product'].value_counts(dropna=False, normalize=True)

# train_data['Credit_Product'] = train_data['Credit_Product'].apply(lambda x : 0 if x==0 else 1)
# test_data['Credit_Product'] = test_data['Credit_Product'].apply(lambda x : 0 if x==0 else 1)

train_data['Credit_Product'].value_counts(dropna=False, normalize=True)
test_data['Credit_Product'].value_counts(dropna=False, normalize=True)

train_data.head()
test_data.head()


# Preparing Data for Modelling

x = train_data.drop(columns='Is_Lead')
y = train_data['Is_Lead']
test = test_data

cat_cols = [c for c in x.columns if c not in num_cols]
cat_cols


# - Feature selection using sklearn.feature_selection.RFE (Recursive Feature Elimnation) was attempted and it selected 5 features but this drastically reduced the ROC AUC Scores
# - Oversampling also decreased the scores

# Train Test Split

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=1)

# Scaling
# - Standard scaling gave better results than MinMax scaling

sc = StandardScaler()
xtrain[num_cols] = sc.fit_transform(xtrain[num_cols])
xtest[num_cols] = sc.transform(xtest[num_cols])
test[num_cols] = sc.transform(test[num_cols])


# Classification - Modelling and Evaluation

# function to calculate ROC AUC score and plot the roc curve
def evaluate_roc(model, name, display=True):
    yprob = model.predict_proba(xtest)[:,1]
    roc_auc = metrics.roc_auc_score(ytest,yprob)

    if display:
        print('ROC AUC =', roc_auc)
        
        # ROC curve
        fpr,tpr,threshold = metrics.roc_curve(ytest,yprob)
        lw = 2
        plt.plot(fpr,tpr,color='darkorange',lw=lw,label='ROC Curve (area = %0.2f)'%roc_auc)
        plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.show()
        
    return {'name':name, 'roc_auc':roc_auc,
           'roc_curve': {'fpr':fpr, 'tpr':tpr, 'threhsold':threshold}}

# function to generate prediction probabilities for test data
def predict_and_export(model, name):
    test_prob = model.predict_proba(test)[:,1]
    df = pd.DataFrame({'ID':test.index,'Is_Lead':test_prob})
    
    filename = 'submissions/submission-' + name + '.csv'
    df.to_csv(filename,index=False)


# Applying different Classification algorithms:

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(xtrain, ytrain)
logreg_roc = evaluate_roc(logreg, 'Logistic Regression')

predict_and_export(logreg, 'logreg')

# Random Forest
rf = RandomForestClassifier()
rf.fit(xtrain, ytrain)
rf_roc = evaluate_roc(rf, 'Random Forest')

predict_and_export(rf, 'rforest')

# Gradient Boosting
gb = GradientBoostingClassifier()
gb.fit(xtrain, ytrain)
gb_roc = evaluate_roc(gb, 'Gradient Boost')

predict_and_export(gb, 'gboost')

# XGBoost
xgb = XGBClassifier(objective='binary:logistic', use_label_encoder=False)
xgb.fit(xtrain, ytrain, eval_metric='auc')
xgb_roc = evaluate_roc(xgb, 'XGBoost')

predict_and_export(xgb, 'xgboost')

# LightGBM
lgbm = LGBMClassifier()
lgbm.fit(xtrain, ytrain)
lgbm_roc = evaluate_roc(lgbm, 'LGBM')

predict_and_export(lgbm, 'lgbm')

pd.DataFrame({'Feature':xtrain.columns, 
              'Importance':lgbm.feature_importances_}).sort_values(by='Importance', ascending=False)

# CatBoost
cb = CatBoostClassifier(verbose=False)
cb.fit(xtrain, ytrain)
cb_roc = evaluate_roc(cb, 'CatBoost')

predict_and_export(cb, 'cb')


# Compare Models
# - Plotting all the ROC courves on one graph to compare them

model_rocs = [logreg_roc, rf_roc, gb_roc, xgb_roc, lgbm_roc, cb_roc]

fig = plt.figure(figsize=(12,9))
lw = 2
plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curves')

for x in model_rocs:
    roc_label = x['name'] + ' (' + str(round(x['roc_auc'], 3)) + ')'
    plt.plot(x['roc_curve']['fpr'], x['roc_curve']['tpr'], lw=lw, label=roc_label)

plt.legend(loc='lower right')
plt.show()


# - LGBMClassifier gave the best ROC AUC score
