🏠 House Price Prediction using XGBoost & Scikit-Learn

This project builds a high-performance regression pipeline for predicting house prices using advanced feature engineering, robust preprocessing, and XGBoost regression, inspired by the Kaggle House Prices: Advanced Regression Techniques competition.

📌 Project Overview

    ~ Goal: Predict house sale prices accurately using structured tabular data

    ~ Dataset: Kaggle House Prices (train.csv, test.csv)

    ~ Target Variable: SalePrice (log-transformed)

    ~ Evaluation Metric: RMSE (Root Mean Squared Error)

🚀 Key Highlights

  -> Extensive feature engineering based on domain knowledge

  -> Outlier removal for better generalization

  -> Automatic numerical & categorical preprocessing

  -> Log transformation for skewed features and target

  -> XGBoost Regressor with strong regularization

  -> Clean train / validation split

  ->Ready-to-use Kaggle submission pipeline

#----------------------------competion scores--------------------------------
---------kaggle score{0.13702} and rank = 1693 ------(1)

model=XGBRegressor(
                      learning_rate=0.06,
                      reg_alpha=0.1,
                      reg_lambda=1
                  )
train rmse is 0.03816104551652763
Cv rmse : 0.1381029254856145
[126026.336 157191.16  183501.88  ... 151739.75  113576.82  210940.38 ]

--------------kaggle score{0.13171} and rank = 1264 ------(2_)

model=XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.06,
                    max_depth=3,
                    reg_alpha=0.1,
                    reg_lambda=1,
                    random_state=42
                  )
train rmse is 0.0712627634882685
Cv rmse : 0.12453284247231235
[121023.78 149638.75 182945.5  ... 159009.36 121764.37 221721.14]

--------------kaggle score{0.13018} and rank = 1950 ------(2_)
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
train rmse is 0.06619626200530847
Cv rmse : 0.122791361417457
[118663.26 160932.52 182607.34 ... 159601.11 118301.79 216122.16]
