from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
#-------------Feature Engineering----------
data=pd.read_csv(r'KaggleHousePricePrediction\data\train.csv')
def feature_engineering(data,train):
    drops=['Id','2ndFlrSF','1stFlrSF','TotalBsmtSF','FullBath','HalfBath','BsmtFullBath','BsmtHalfBath','OpenPorchSF','ScreenPorch','3SsnPorch','EnclosedPorch','OverallQual','OverallCond','YrSold','YearRemodAdd','YearBuilt']
    
    data['TotalBathrooms']=data['FullBath']+0.5*data['HalfBath']+data['BsmtFullBath']+0.5*data['BsmtHalfBath']

    data['TotalPorchSF']=data['OpenPorchSF']+data['ScreenPorch']+data['3SsnPorch']+data['EnclosedPorch']

    data['TotalSF']=data['2ndFlrSF']+data['1stFlrSF']+data['TotalBsmtSF']

    data['OverallScore']=data['OverallQual']*data['OverallCond']

    data['AgeAtSale']=data['YrSold']-data['YearBuilt']

    data['RemodAge']=data['YrSold']-data['YearRemodAdd']
    data.drop(columns=drops,inplace=True,errors='ignore')
    if train:
        data['SalePrice']=np.log1p(data['SalePrice'])
        data=data.drop(data[(data['GrLivArea']>4000)&(data['SalePrice']<300000)].index)
    features=data.select_dtypes([np.number]).columns
    skew=data[features].apply(lambda x:x.skew()).sort_values(ascending=False)
    skewedfeatures=['MiscVal','PoolArea','LotArea','LowQualFinSF','KitchenAbvGr','BsmtFinSF2','MasVnrArea','TotalPorchSF','WoodDeckSF','LotFrontage','MSSubClass','BsmtUnfSF','GrLivArea']
    for col in skewedfeatures:
        data[col]=np.log1p(data[col])
    return data
data=feature_engineering(data,train=True)
X=data.drop(['SalePrice'],axis=1)
Y=data['SalePrice']
x_train,x_cv,y_train,y_cv=train_test_split(X,Y,test_size=0.2,random_state=42)
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
x_cv=preprocessor.fit_transform(x_cv)
model=XGBRegressor(n_estimators=500,learning_rate=0.06,max_depth=6,min_child_weight=1,gamma=0,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.3,reg_lambda=1.8,objective='reg:squarederror',eval_metric='rmse',random_state=42)
#model=RandomForestRegressor(n_estimators=400,max_depth=6,)

model_=Pipeline(steps=[
    ('model',model)
])

model_.fit(x_train,y_train,model__eval_set=[(x_cv,y_cv)],model__early_stopping_rounds=50)

pred=model_.predict(x_train)
predcv=model_.predict(x_cv)
tmse=np.sqrt(mean_squared_error(y_train,pred))
cvmse=np.sqrt(mean_squared_error(y_cv,predcv))
print(f'train rmse is {tmse}')
print(f'Cv rmse : {cvmse}')


#--------------testing----------
testdata=pd.read_csv(r'KaggleHousePricePrediction\data\test.csv')
test_ids=testdata['Id']
testdata=feature_engineering(testdata,train=False)
pred_test=model_.predict(testdata)

prices=np.expm1(pred_test)
print(prices)
s=pd.DataFrame({'Id':test_ids,'SalePrice':prices})
s.to_csv('submission1.csv',index=False)