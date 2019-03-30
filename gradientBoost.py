import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
from scipy import stats
from scipy.stats import skew #for some statistics
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor

dirname = os.path.dirname(os.path.abspath(__file__))
train_file = os.path.join(dirname, 'input/train.csv')
test_file = os.path.join(dirname, 'input/test.csv')
train = pd.read_csv(train_file)
test = pd.read_csv(test_file)

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

#Deleting outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

train["SalePrice"] = np.log1p(train["SalePrice"])

train_ID_withDrop = train['Id']

#Now drop the  'Id' colum since it's unnecessary for  the prediction process.
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#Drop Outlier column from our investigation
train.drop("MiscVal", axis = 1, inplace = True)
test.drop("MiscVal", axis = 1, inplace = True)
train.drop("MiscFeature", axis = 1, inplace = True)
test.drop("MiscFeature", axis = 1, inplace = True)
train.drop("Utilities", axis = 1, inplace = True)
test.drop("Utilities", axis = 1, inplace = True)

ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)

all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
all_data['OverallCond'] = all_data['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

# Adding total sqfootage feature 
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

all_data = pd.get_dummies(all_data)
print(all_data.shape)

train = all_data[:ntrain]
test = all_data[ntrain:]

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
GBoost.fit(train.values, y_train)
train_pred = GBoost.predict(train.values)


execTime = []
for i in range(10) :
    start = time.time()
    test_pred = np.expm1(GBoost.predict(test.values))
    end = time.time()
    execTime.append(end - start)

averageExecTime = sum(execTime) / len(execTime)

print("AverageExecutionTime : ", averageExecTime)

salePriceGap = []
maxPriceGap = 0
minPriceGap = np.expm1(train_pred[0])
pctGap = []
for i in range(len(train_pred)) :
    value = abs(np.expm1(y_train[i]) - np.expm1(train_pred[i]))
    pctGap.append(value / np.expm1(y_train[i]))
    salePriceGap.append(value)

    if value < minPriceGap:
        minPriceGap = value

    if value > maxPriceGap:
        maxPriceGap = value

# print(pctGap)
averagePriceGap = np.average(salePriceGap)
averagePctGap = np.average(pctGap)

print("AveragePctGap : {}".format(averagePctGap))

print("AveragePriceGap : {} \nMaxPriceGap : {} \nMinPriceGap : {}".format(averagePriceGap, maxPriceGap, minPriceGap))

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

print(rmsle(y_train, train_pred))

sub = pd.DataFrame()
sub['Id'] = test_ID
sub['SalePrice'] = test_pred
sub.to_csv('submissionGBoost.csv',index=False)

gap = pd.DataFrame()
gap['Id'] = train_ID_withDrop
gap['pctGap'] = pctGap
gap['realSalePrice'] = np.expm1(y_train)
gap['predictedSalePrice'] = np.expm1(train_pred)
gap = gap[gap['pctGap'] > 0.25]
gap.to_csv('pctgapGBoost.csv',index=False)