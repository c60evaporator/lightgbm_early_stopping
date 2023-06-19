# %% Scikit-Learn API + 旧指定方法(early_stopping_rounds) + cross_val_score_eval_setメソッドでクロスバリデーション
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
import lightgbm as lgb
from sklearn.model_selection import KFold
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
        'random_state': 42,  # 乱数シード
        'boosting_type': 'gbdt',  # boosting_type
        'n_estimators': 10000  # 最大学習サイクル数。early_stopping使用時は大きな値を入力
        }
verbose_eval = 0  # この数字を1にすると学習時のスコア推移がコマンドライン表示される
# early_stoppingを指定してLightGBM学習
lgbr = lgb.LGBMRegressor(**params)
# クロスバリデーション内部で`fit()`メソッドに渡すパラメータ
fit_params = {'eval_metric':'rmse',
              'eval_set':[(X, y)],
              'early_stopping_rounds': 10,
              'verbose': verbose_eval}
# クロスバリデーション実行
scores = cross_val_score_eval_set(
        eval_set_selection='test',  # 'test'と指定するとテストデータを'eval_set'に渡せる
        estimator=lgbr,  # 学習器
        X=X, y=y,  # クロスバリデーション分割前のデータを渡す
        scoring='neg_root_mean_squared_error',  # RMSE（の逆数）を指定
        cv=cv, verbose=verbose_eval, fit_params=fit_params
        )
print(f'RMSE={scores} \nRMSE mean={np.mean(scores)}')

# %%
