#%%
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold
import lightgbm as lgb
from seaborn_analyzer import cross_val_score_eval_set

# データセット読込(カリフォルニア住宅価格)
TARGET_VARIABLE = 'price'  # 目的変数名
USE_EXPLANATORY = ['MedInc', 'AveOccup', 'Latitude', 'HouseAge']  # 説明変数名
california_housing = pd.DataFrame(np.column_stack((fetch_california_housing().data, fetch_california_housing().target)),
    columns = np.append(fetch_california_housing().feature_names, TARGET_VARIABLE))
california_housing = california_housing.sample(n=1000, random_state=42)  # データ数多いので1000にサンプリング
y = california_housing[TARGET_VARIABLE].values  # 目的変数のnumpy配列
X = california_housing[USE_EXPLANATORY].values  # 説明変数のnumpy配列

# クロスバリデーション用のScikit-Learnクラス（5分割KFold）
cv = KFold(n_splits=5, shuffle=True, random_state=42)

###### ここからがLightGBMの実装 ######
# 使用するパラメータ
params = {'objective': 'regression',  # 最小化させるべき損失関数
          'metric': 'rmse',  # 学習時に使用する評価指標(early_stoppingの評価指標にも同じ値が使用される)
          'random_state': 42,  # 乱数シード
          'boosting_type': 'gbdt',  # boosting_type
          'n_estimators': 10000,  # 最大学習サイクル数。early_stopping使用時は大きな値を入力
          'verbose': -1,  # これを指定しないと`No further splits with positive gain, best gain: -inf`というWarningが表示される
          'early_stopping_round': 10  # ここでearly_stoppingを指定
          }
# early_stoppingを指定してLightGBM学習
lgbr = lgb.LGBMRegressor(**params)
# クロスバリデーション内部で`fit()`メソッドに渡すパラメータ
fit_params = {'eval_set':[(X, y)]
              }
# クロスバリデーション実行
scores = cross_val_score_eval_set(
        validation_fraction=0.3,  # floatで指定した割合で学習データから'eval_set'を分割する（同時に学習データからeval_setの分が除外される）
        estimator=lgbr,  # 学習器
        X=X, y=y,  # クロスバリデーション分割前のデータを渡す
        scoring='neg_root_mean_squared_error',  # RMSE（の逆数）を指定
        cv=cv, fit_params=fit_params
        )
print(f'RMSE={scores} \nRMSE mean={np.mean(scores)}')
# %%
