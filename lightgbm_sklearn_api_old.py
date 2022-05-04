# %% Scikit-Learn API + 旧指定方法(early_stopping_rounds)での実装法
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
# データセット読込(カリフォルニア住宅価格)
TARGET_VARIABLE = 'price'  # 目的変数名
USE_EXPLANATORY = ['MedInc', 'AveOccup', 'Latitude', 'HouseAge']  # 説明変数名
california_housing = pd.DataFrame(np.column_stack((fetch_california_housing().data, fetch_california_housing().target)),
    columns = np.append(fetch_california_housing().feature_names, TARGET_VARIABLE))
california_housing = california_housing.sample(n=1000, random_state=42)  # データ数多いので1000にサンプリング
y = california_housing[TARGET_VARIABLE].values  # 目的変数のnumpy配列
X = california_housing[USE_EXPLANATORY].values  # 説明変数のnumpy配列
# テストデータと学習データ分割
X_train_raw, X_test, y_train_raw, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# early_stopping用の評価データをさらに分割
X_train, X_valid, y_train, y_valid = train_test_split(X_train_raw, y_train_raw, test_size=0.25, random_state=42)

###### ここからがLightGBMの実装 ######
# 使用するパラメータ
params = {'objective': 'regression',  # 最小化させるべき損失関数
         'random_state': 42,  # 乱数シード
         'boosting_type': 'gbdt',  # boosting_type
         'n_estimators': 10000  # 最大学習サイクル数。early_stopping使用時は大きな値を入力
         }
verbose_eval = 0  # この数字を1にすると学習時のスコア推移がコマンドライン表示される
# early_stoppingを指定してLightGBM学習
lgbr = lgb.LGBMRegressor(**params)
lgbr.fit(X_train, y_train, 
         eval_metric='rmse',  # early_stoppingの評価指標(学習用の'metric'パラメータにも同じ指標が自動入力される)
         eval_set=[(X_valid, y_valid)],
         early_stopping_rounds=10,
         verbose=verbose_eval
         )

# スコア(RMSE)算出
y_pred = lgbr.predict(X_test)
score = mean_squared_error(y_true=y_test, y_pred=y_pred, squared=False)
print(f'RMSE={score}')

# %%
