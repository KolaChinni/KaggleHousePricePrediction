from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,KFold
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor
from xgboost.callback import EarlyStopping
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
#-------------Feature Engineering----------
data=pd.read_csv(r'KaggleHousePricePrediction(#)\data\train.csv')

def feature_engineering(data):
    drops=['Id','2ndFlrSF','1stFlrSF','TotalBsmtSF','FullBath','HalfBath','BsmtFullBath','BsmtHalfBath','OpenPorchSF','ScreenPorch','3SsnPorch','EnclosedPorch','YrSold','YearRemodAdd','YearBuilt']
    
    data['TotalBathrooms']=data['FullBath']+0.5*data['HalfBath']+data['BsmtFullBath']+0.5*data['BsmtHalfBath']

    data['TotalPorchSF']=data['OpenPorchSF']+data['ScreenPorch']+data['3SsnPorch']+data['EnclosedPorch']

    data['TotalSF']=data['2ndFlrSF']+data['1stFlrSF']+data['TotalBsmtSF']

    data['OverallScore']=data['OverallQual']*data['OverallCond']

    data['AgeAtSale']=data['YrSold']-data['YearBuilt']

    data['RemodAge']=data['YrSold']-data['YearRemodAdd']

    data['QualSF']=data['OverallQual']*data['GrLivArea']

    data['GarageScore']=data['GarageCars']*data['GarageArea']

    data['BsmtScore']=data['TotalBsmtSF']*data['BsmtQual'].map({'Ex':5,'Gd':4,'TA':3,'Fa':2,'Po':1,'NA':0})

    data['totalrooms']=data['TotRmsAbvGrd']+data['BedroomAbvGr']


    data.drop(columns=drops,inplace=True,errors='ignore')
    features=data.select_dtypes([np.number]).columns
    skew=data[features].apply(lambda x:x.skew()).sort_values(ascending=False)
    skewedfeatures=['MiscVal','PoolArea','LotArea','LowQualFinSF','KitchenAbvGr','BsmtFinSF2','MasVnrArea','TotalPorchSF','WoodDeckSF','LotFrontage','MSSubClass','BsmtUnfSF','GrLivArea']
    for col in skewedfeatures:
        if col in data.columns:
            data[col]=np.log1p(data[col])
    return data
outlier_mask = ~((data['GrLivArea'] > 4000) & (data['SalePrice'] < 300000))
data = data[outlier_mask]
X=data.drop(['SalePrice'],axis=1)
Y=data['SalePrice']
Y=np.log1p(Y)
X=feature_engineering(X)
x_train,x_cv,y_train,y_cv=train_test_split(X,Y,test_size=0.2,random_state=42)

#-------------cleaning and processing----------
numerical_features=x_train.select_dtypes(['float64','int64']).columns
categorrical_features=x_train.select_dtypes(['object']).columns

num_transformer=Pipeline(steps=[
   ('imputer',SimpleImputer(strategy='median'))
])
cat_transformer=Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('encoder',OneHotEncoder(handle_unknown='ignore'))
])

preprocessor=ColumnTransformer(
    transformers=[
        ('num',num_transformer,numerical_features),
        ('cat',cat_transformer,categorrical_features)
    ]
)
x_train=preprocessor.fit_transform(x_train)
x_cv=preprocessor.transform(x_cv)
#-------------Model--------------
model_=XGBRegressor(
                        n_estimators=5000,
                        learning_rate=0.01,
                        max_depth=3,
                        min_child_weight=9,
                        gamma=0,
                        subsample=0.6,
                        colsample_bytree=0.8,
                        reg_alpha=0.8,
                        reg_lambda=20.0,
                        eval_metric='rmse',
                        objective='reg:squarederror',
                        random_state=42
                    )
#model=RandomForestRegressor(n_estimators=400,max_depth=6,)



model_.fit(x_train,y_train,eval_set=[(x_cv,y_cv)])

pred=model_.predict(x_train)
predcv=model_.predict(x_cv)
tmse=np.sqrt(mean_squared_error(y_train,pred))
cvmse=np.sqrt(mean_squared_error(y_cv,predcv))
print(f'train rmse is {tmse}')
print(f'Cv rmse : {cvmse}')


#--------------testing----------
'''
testdata=pd.read_csv(r'KaggleHousePricePrediction(#)\data\test.csv')
test_ids=testdata['Id']
testdata=feature_engineering(testdata)
x_test=preprocessor.transform(testdata)
pred_test=model_.predict(x_test)

prices=np.expm1(pred_test)
print(prices)
s=pd.DataFrame({'Id':test_ids,'SalePrice':prices})
s.to_csv('submission1.csv',index=False) '''
