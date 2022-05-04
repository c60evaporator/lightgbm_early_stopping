# %% Training API + 新指定方法(コールバック関数early_stopping) + lightgbm.cv()メソッドのクロスバリデーション
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.model_selection import KFold
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
# データをDatasetクラスに格納
dcv = lgb.Dataset(X, label=y)  # クロスバリデーション用
# 使用するパラメータ
params = {'objective': 'regression',  # 最小化させるべき損失関数
         'metric': 'rmse',  # 学習時に使用する評価指標(early_stoppingの評価指標にも同じ値が使用される)
         'random_state': 42,  # 乱数シード
         'boosting_type': 'gbdt',  # boosting_type
         'verbose': -1  # これを指定しないと`No further splits with positive gain, best gain: -inf`というWarningが表示される
         }
verbose_eval = 0  # この数字を1にすると学習時のスコア推移がコマンドライン表示される
# early_stoppingを指定してLightGBMをクロスバリデーション
cv_result = lgb.cv(params, dcv,
                num_boost_round=10000,  # 最大学習サイクル数。early_stopping使用時は大きな値を入力
                folds=cv,
                callbacks=[lgb.early_stopping(stopping_rounds=10, 
                                verbose=True), # early_stopping用コールバック関数
                           lgb.log_evaluation(verbose_eval)] # コマンドライン出力用コールバック関数
                )
print(cv_result)
print(f'RMSE mean={cv_result["rmse-mean"][-1]}')
# %%
