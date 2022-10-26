import gc
gc.collect()

import time
import pandas as pd
import numpy as np
import seaborn as sns
#sns.set(rc={'figure.figsize':(20,10)})
import matplotlib.pyplot as plt
import datetime

from sklearn.preprocessing import LabelEncoder,OneHotEncoder, StandardScaler, MinMaxScaler
lbec = LabelEncoder()
scaler = StandardScaler()

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, plot_precision_recall_curve
import xgboost as xgb
from xgboost import plot_tree
from xgboost import plot_tree, XGBClassifier, plot_importance
import lightgbm as lgb
from lightgbm import plot_importance as lgbm_plt
#from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from imblearn.over_sampling import SMOTE
# import pycaret
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
import re
import pickle
import os
os.chdir(r"C:\Users\andyhlso\Desktop\Andy\Kaggle\202209 - house prices")

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
df2 = df.copy()
df2 = df2.set_index('Id')
df2_test = df_test.copy()
df2_test = df2_test.set_index('Id')

df_all = pd.concat([df2,df2_test])

df_all.columns
df_all.dtypes

# drop_vars = ['Name']
# df2 = df2.drop(drop_vars, axis=1)
# df2_test = df2_test.drop(drop_vars, axis=1)

df2['SalePrice'].isna().sum()


"""======================================== Data cleaning ========================================"""
# def check_missing_row(missing_rate):   
#     missing_row = (df2.isnull().sum(axis=1))/len(df2.columns) *100
#     chk = df2.loc[missing_row[missing_row >= missing_rate].index]
#     print('Missing rows pct > ' + str(missing_rate) + ' by Group:')
#     print(chk.value_counts())

# def remove_missing_row_data(df, missing_rate):
#     #remove data with 60% missing
#     #most of missing rows are inactive customers, so remove them
#     #check missing value by rows
#     missing_row = (df.isnull().sum(axis=1))/len(df.columns) *100
#     return(df.loc[missing_row[missing_row < missing_rate].index])


def check_missing_col(df, threshold):
    #check missing values by column
    missing = df.isnull().sum()/ len(df) *100
    missing = missing[missing > threshold]
    return(missing)

def remove_missing_col_data(df, missing_rate):
    missing = df_all.isnull().sum()/ len(df_all) *100
    missing = missing[missing > 0]
    #fill 0 withs missing percentage greater than 80
    missing80pct_cols = missing[missing > missing_rate].index.tolist()
    #df[missing80pct_cols] = df[missing80pct_cols].fillna(0)
    print('no of cols to drop: ' + str(len(missing80pct_cols)))
    return(df.drop(missing80pct_cols, axis=1))

check_missing_col(df_all, 10)
# LotFrontage    16.649538
# Alley          93.216855
# FireplaceQu    48.646797
# PoolQC         99.657417
# Fence          80.438506
# MiscFeature    96.402878
# SalePrice      49.982871

df_all['Alley'].value_counts(dropna=False).reset_index()
df_all['FireplaceQu'].value_counts(dropna=False).reset_index()
df_all['PoolQC'].value_counts(dropna=False).reset_index()
df_all['Fence'].value_counts(dropna=False).reset_index()
df_all['MiscFeature'].value_counts(dropna=False).reset_index()

none_vars = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']
for var in none_vars:
    df2[var] = df2[var].fillna('None')
    df2_test[var] = df2_test[var].fillna('None')

df2['LotFrontage'] = df2['LotFrontage'].fillna(df_all['LotFrontage'].mean())
df2_test['LotFrontage'] = df2_test['LotFrontage'].fillna(df_all['LotFrontage'].mean())

check_missing_col(df2, 0)
# MasVnrType      0.547945
# MasVnrArea      0.547945
# BsmtQual        2.534247
# BsmtCond        2.534247
# BsmtExposure    2.602740
# BsmtFinType1    2.534247
# BsmtFinType2    2.602740
# Electrical      0.068493
# GarageType      5.547945
# GarageYrBlt     5.547945
# GarageFinish    5.547945
# GarageQual      5.547945
# GarageCond      5.547945

df_all['Electrical'].value_counts()
df2['Electrical'] = df2['Electrical'].fillna('SBrkr')
df2_test['Electrical'] = df2_test['Electrical'].fillna('SBrkr')

chk = df_all[['MasVnrType','MasVnrArea','BsmtQual','BsmtCond','BsmtExposure',
       'BsmtFinType1','BsmtFinType2','GarageType','GarageYrBlt','GarageFinish','GarageQual','GarageCond']].head(20)

none2_vars = ['MasVnrType','BsmtQual','BsmtCond','BsmtExposure',
       'BsmtFinType1','BsmtFinType2','GarageType','GarageFinish','GarageQual','GarageCond']
for var in none2_vars:
    df2[var] = df2[var].fillna('None')
    df2_test[var] = df2_test[var].fillna('None')

df2['MasVnrArea'] = df2['MasVnrArea'].fillna(0)
df2_test['MasVnrArea'] = df2_test['MasVnrArea'].fillna(0)

######################################MISSING VALUE DONE####################################
check_missing_col(df2, 0)
chk = df_all['GarageYrBlt'].value_counts(dropna=False)
df_all['GarageYrBlt'].max()

df2['GarageTenure'] = 2022 - df2[df2['GarageYrBlt']<=2022]['GarageYrBlt']
df2['GarageTenure'] = df2['GarageTenure'].fillna(199)
df2 = df2.drop('GarageYrBlt',axis=1)
df2_test['GarageTenure'] = 2022 - df2_test[df2_test['GarageYrBlt']<=2022]['GarageYrBlt']
df2_test['GarageTenure'] = df2_test['GarageTenure'].fillna(199)
df2_test = df2_test.drop('GarageYrBlt',axis=1)

df2['HouseTenure'] = 2022 - df2[df2['YearBuilt']<=2022]['YearBuilt']
df2['HouseTenure'] = df2['HouseTenure'].fillna(199)
df2 = df2.drop('YearBuilt',axis=1)
df2_test['HouseTenure'] = 2022 - df2_test[df2_test['YearBuilt']<=2022]['YearBuilt']
df2_test['HouseTenure'] = df2_test['HouseTenure'].fillna(199)
df2_test = df2_test.drop('YearBuilt',axis=1)

df2['HouseRemodTenure'] = 2022 - df2[df2['YearRemodAdd']<=2022]['YearRemodAdd']
df2['HouseRemodTenure'] = df2['HouseRemodTenure'].fillna(199)
df2 = df2.drop('YearRemodAdd',axis=1)
df2_test['HouseRemodTenure'] = 2022 - df2_test[df2_test['YearRemodAdd']<=2022]['YearRemodAdd']
df2_test['HouseRemodTenure'] = df2_test['HouseRemodTenure'].fillna(199)
df2_test = df2_test.drop('YearRemodAdd',axis=1)

df2['SoldTenure'] = 2022 - df2[df2['YrSold']<=2022]['YrSold']
df2['SoldTenure'] = df2['SoldTenure'].fillna(199)
df2 = df2.drop('YrSold',axis=1)
df2_test['SoldTenure'] = 2022 - df2_test[df2_test['YrSold']<=2022]['YrSold']
df2_test['SoldTenure'] = df2_test['SoldTenure'].fillna(199)
df2_test = df2_test.drop('YrSold',axis=1)



df2.columns

map1 = {'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1}
df2['LotShape'] = df2['LotShape'].map(map1)
df2_test['LotShape'] = df2_test['LotShape'].map(map1)

map2 = {'Lvl':4, 'Bnk':3, 'HLS':2, 'Low':1}
df2['LandContour'] = df2['LandContour'].map(map2)
df2_test['LandContour'] = df2_test['LandContour'].map(map2)

map3 = {'AllPub':4, 'NoSewr':3, 'NoSeWa':2, 'ELO':1}
df2['Utilities'] = df2['Utilities'].map(map3)
df2_test['Utilities'] = df2_test['Utilities'].map(map3)

map4 = {'Gtl':3, 'Mod':2, 'Sev':1}
df2['LandSlope'] = df2['LandSlope'].map(map4)
df2_test['LandSlope'] = df2_test['LandSlope'].map(map4)

map5 = {'1Fam':5, '2FmCon':4, 'Duplx':3, 'TwnhsE':2, 'TwnhsI':1}
df2['BldgType'] = df2['BldgType'].map(map5)
df2_test['BldgType'] = df2_test['BldgType'].map(map5)

map6 = {'SLvl':8, 'SFoyer':7, '2.5Unf':6, '2.5Fin':5, '2Story':4, '1.5Unf':3, '1.5Fin':2, '1Story':1}
df2['HouseStyle'] = df2['HouseStyle'].map(map6)
df2_test['HouseStyle'] = df2_test['HouseStyle'].map(map6)

map7 = {'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'None':0, 'NA':0}
qual_vars = ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual',
             'GarageCond','PoolQC']
for var in qual_vars:
    df2[var] = df2[var].map(map7)
    df2_test[var] = df2_test[var].map(map7)
    
map8 = {'Gd':5, 'Av':4, 'Mn':3, 'No':2, 'None':1, 'NA':0}
df2['BsmtExposure'] = df2['BsmtExposure'].map(map8)
df2_test['BsmtExposure'] = df2_test['BsmtExposure'].map(map8)

map9 = {'GLQ':7, 'ALQ':6, 'BLQ':5, 'Rec':4, 'LwQ':3, 'Unf':2, 'None':1, 'NA':1}
fin_vars = ['BsmtFinType1','BsmtFinType2']
for var in fin_vars:
    df2[var] = df2[var].map(map9)
    df2_test[var] = df2_test[var].map(map9)

map10 = {'Y':1,'N':0}
df2['CentralAir'] = df2['CentralAir'].map(map10)
df2_test['CentralAir'] = df2_test['CentralAir'].map(map10)

map11 = {'Typ':8, 'Min1':7, 'Min2':6, 'Mod':5, 'Maj1':4, 'Maj2':3, 'Sev':2, 'Sal':1}
df2['Functional'] = df2['Functional'].map(map11)
df2_test['Functional'] = df2_test['Functional'].map(map11)

map12 = {'Fin':4, 'RFn':3, 'Unf':2, 'None':1, 'NA':1}
df2['GarageFinish'] = df2['GarageFinish'].map(map12)
df2_test['GarageFinish'] = df2_test['GarageFinish'].map(map12)

map13 = {'Y':3, 'P':2, 'N':1}
df2['PavedDrive'] = df2['PavedDrive'].map(map13)
df2_test['PavedDrive'] = df2_test['PavedDrive'].map(map13)



chk = df2.dtypes

condition_vars = np.unique(df_all[['Condition1','Condition2']].values)
for var in condition_vars:
    df2['cond_'+ var] = df2.apply(lambda row: (1 if var == row.Condition1 else 0) +
                                              (1 if var == row.Condition2 else 0) ,axis=1)
    df2_test['cond_'+ var] = df2_test.apply(lambda row: (1 if var == row.Condition1 else 0) +
                                                        (1 if var == row.Condition2 else 0) ,axis=1)
df2 = df2.drop(['Condition1','Condition2'],axis=1)
df2_test = df2_test.drop(['Condition1','Condition2'],axis=1)


exterior_vars = np.unique(df_all[['Exterior1st','Exterior2nd']].astype(str).values)
for var in exterior_vars:
    df2['ext_'+ var] = df2.apply(lambda row: (1 if var == str(row.Exterior1st) else 0) +
                                              (1 if var == str(row.Exterior2nd) else 0) ,axis=1)
    df2_test['ext_'+ var] = df2_test.apply(lambda row: (1 if var == str(row.Exterior1st) else 0) +
                                                        (1 if var == str(row.Exterior2nd) else 0) ,axis=1)
df2 = df2.drop(['Exterior1st','Exterior2nd'],axis=1)
df2_test = df2_test.drop(['Exterior1st','Exterior2nd'],axis=1)

chk = [k for k in df2.columns if 'ext' in k]

one_hot_vars = ['MSSubClass','MSZoning','Street','Alley', 'LotConfig', 'Neighborhood','RoofStyle', 'RoofMatl', 
                'MasVnrType','Foundation','Heating','Electrical','GarageType','MiscFeature','Fence','SaleType',
                'SaleCondition']
# 17
label_vars = ['LotShape','LandContour','Utilities','LandSlope','OverallQual', 'OverallCond','cond_Feedr', 'cond_Norm',
              'cond_PosA', 'cond_Artery','cond_PosN', 'cond_RRAe','cond_RRAn', 'cond_RRNe', 'cond_RRNn','BldgType',
              'ext_AsbShng', 'ext_AsphShn', 'ext_Brk Cmn', 'ext_BrkComm', 'ext_BrkFace', 'ext_CBlock', 'ext_CemntBd',
              'ext_CmentBd', 'ext_HdBoard', 'ext_ImStucc', 'ext_MetalSd', 'ext_Other', 'ext_Plywood', 'ext_Stone', 
              'ext_Stucco', 'ext_VinylSd', 'ext_Wd Sdng', 'ext_Wd Shng', 'ext_WdShing', 'ext_nan','HouseStyle',
              'ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu','GarageQual',
              'GarageCond','PoolQC','BsmtExposure','BsmtFinType1','BsmtFinType2','CentralAir','Functional',
              'GarageFinish','PavedDrive','BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 
              'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars','MoSold']
# 64
continuous_vars = ['LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
                   '2ndFlrSF', 'LowQualFinSF', 'GrLivArea','GarageTenure', 'HouseTenure', 'HouseRemodTenure',
                   'SoldTenure','TotalBsmtSF', '1stFlrSF','WoodDeckSF', 'OpenPorchSF','PoolArea','GarageArea',
                   'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal']
# 23
# 104 total

chk = [k for k in df2_test.columns if k not in one_hot_vars and k not in label_vars and k not in continuous_vars]
chk2 = df2_test[chk].head(20)

# np.unique(continuous_vars,return_counts=True)

df2['train_test'] = 'train'
df2_test['train_test'] = 'test'
df2_all = pd.concat([df2,df2_test])

for var in one_hot_vars:
    df2_all = pd.get_dummies(df2_all, columns=[var], prefix = ['OH_' + var])

df2_all[continuous_vars] = scaler.fit_transform(df2_all[continuous_vars])

for var in label_vars:   
    df2_all['label_'+var] = lbec.fit_transform(df2_all[var].astype(str))
        
df2_all = df2_all.drop(label_vars,axis=1)


corr_df = df2_all.corr().abs()

def remove_correlated(df):
    corr_df = df.corr().abs()
    # Create and apply mask
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    tri_df = corr_df.mask(mask)
    tri_df.to_excel('df_correlation.xlsx')
    # Find columns that meet treshold
    to_drop = [c for c in tri_df.columns if any(tri_df[c] > 0.7)]
    print(to_drop)
    #df = df.drop(to_drop, axis=1)
    print("removed correlation >0.7")
    return(df.drop(to_drop, axis=1))

remove_correlated(df2_all)

df2_all = remove_correlated(df2_all)
# ['BsmtFinSF1', 'BsmtFinSF2', 'TotalBsmtSF', 'GrLivArea', 'GarageArea', 'PoolArea', 'GarageTenure', 
# 'OH_MSSubClass_90', 'OH_MSZoning_FV', 'OH_MSZoning_RL', 'OH_Street_Grvl', 'OH_Alley_Grvl', 'OH_LotConfig_Corner',
#  'OH_Neighborhood_NPkVill', 'OH_RoofStyle_Flat', 'OH_RoofStyle_Gable', 'OH_RoofMatl_CompShg', 'OH_MasVnrType_BrkFace',
#  'OH_Foundation_CBlock', 'OH_Heating_GasA', 'OH_Electrical_FuseA', 'OH_GarageType_Attchd', 'OH_GarageType_None', 
#  'OH_MiscFeature_None', 'OH_Fence_MnPrv', 'OH_SaleType_New', 'OH_SaleType_WD', 'label_ext_CemntBd', 'label_ExterQual',
#  'label_FireplaceQu', 'label_GarageQual']
# removed correlation >0.7
df2 = df2_all[df2_all['train_test'] == 'train']
df2_test = df2_all[df2_all['train_test'] == 'test']

df2_test = df2_test.drop('SalePrice',axis=1)
df2 = df2.drop('train_test',axis=1)
df2_test = df2_test.drop('train_test',axis=1)

######################################PRE PROCESSING DONE####################################
#_______modelling_________#
X = df2.drop('SalePrice', axis=1)
y = df2['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

models = []
models.append(("KRR",KernelRidge()))
models.append(("GB",GradientBoostingRegressor()))
models.append(("RF",RandomForestRegressor()))
models.append(("XGB",xgb.XGBRegressor()))
models.append(("LGBM",lgb.LGBMRegressor()))

for name,model in models:
    kfold = KFold(n_splits=5, random_state=22, shuffle=True)
    cv_result = cross_val_score(model,X_train,y_train, cv = kfold,scoring = "r2")
    print(name, cv_result)
    
# KRR [0.8701243  0.77527672 0.85116484 0.83862925 0.45428562]
# GB [0.87123088 0.82639474 0.8862214  0.89148291 0.64790292]
# RF [0.83772038 0.82336631 0.85907404 0.8779928  0.70736905]
# XGB [0.85152234 0.6821586  0.87675909 0.87658629 0.69182037]
# LGBM [0.85278385 0.84838442 0.87961264 0.86708417 0.73598484]
    

# """GBR"""
clf = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
clf.fit(X_train, y_train)

# featureImp= []
# for feat, importance in zip(X_train.columns, clf.feature_importances_):  
#     temp = [feat, importance*100]
#     featureImp.append(temp)
# model_df = pd.DataFrame(featureImp,columns = ['Feature', 'Importance']).sort_values('Importance', ascending = False)
# sns.barplot(x=model_df['Importance'], y=model_df['Feature'])

def feat_imp(df, model, n_features):

    d = dict(zip(df.columns, model.feature_importances_))
    ss = sorted(d, key=d.get, reverse=False)
    top_names = ss[0:n_features]

    plt.figure(figsize=(15,15))
    plt.title("Feature importances")
    plt.barh(range(n_features), [d[i] for i in top_names], align="center")
    plt.ylim(-1, n_features)
    plt.yticks(range(n_features), top_names, rotation='horizontal')
    return(top_names)

feat_imp(X_train, clf, 20)

# feature_importance = clf.feature_importances_
# sorted_idx = np.argsort(feature_importance)
# pos = np.arange(sorted_idx.shape[0]) + 0.5
# fig = plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.barh(pos, feature_importance[sorted_idx], align="center")
# plt.yticks(pos, np.array(X.columns)[sorted_idx])
# plt.title("Feature Importance")

# ax = plot_importance(clf,max_num_features = 20)
# plot_importance(clf)
# fig = ax.figure
# fig.set_size_inches(10, 10)

top_feat = ['label_KitchenQual',
 '1stFlrSF',
 'label_FullBath',
 'HouseTenure',
 'label_GarageCars',
 'label_OverallQual',
 'label_GarageFinish',
 'label_Fireplaces',
 'HouseRemodTenure',
 'label_BsmtQual',
 'LotArea',
 '2ndFlrSF',
 'OH_MSSubClass_60',
 'label_BsmtFinType1',
 'MasVnrArea',
 'label_TotRmsAbvGrd',
 'label_HeatingQC',
 'LotFrontage',
 'label_CentralAir',
 'OpenPorchSF',
 'SalePrice']

df3 = df2[top_feat]
X = df3.drop('SalePrice', axis=1)
y = df3['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
clf.fit(X_train, y_train)

pickle.dump(clf,open('house_price_regression.pkl','wb'))
clf_f = pickle.load(open('house_price_regression.pkl','rb'))

top_feat.remove('SalePrice')
df3_test = df2_test[top_feat]

pred = clf.predict(df3_test)
out = pd.DataFrame({'Id':df3_test.index, 'SalePrice':pred})


out.to_csv('house_price_regression.csv',index=False)

