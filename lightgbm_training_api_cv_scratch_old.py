# %% Training API + 旧指定方法(early_stopping_rounds) + スクラッチ実装のクロスバリデーション
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

# クロスバリデーションのデータ分割
scores=[]
for i, (train, test) in enumerate(cv.split(X, y)):
    ###### ここからがLightGBMの実装 ######
    # データをDatasetクラスに格納
    dtrain = lgb.Dataset(X[train], label=y[train])  # 学習データ
    dvalid = lgb.Dataset(X[test], label=y[test])  # early_stopping用(テストデータを使用)
    # 使用するパラメータ
    param = {'objective': 'regression',  # 最小化させるべき損失関数
             'metric': 'rmse',  # 学習時に使用する評価指標(early_stoppingの評価指標にも同じ値が使用される)
             'random_state': 42,  # 乱数シード
             'boosting_type': 'gbdt',  # boosting_type
             'verbose': -1  # これを指定しないと`No further splits with positive gain, best gain: -inf`というWarningが表示される
             }
    verbose_eval = 0  # この数字を1にすると学習時のスコア推移がコマンドライン表示される
    # early_stoppingを指定してLightGBM学習
    gbm = lgb.train(param, dtrain,
                    valid_sets=[dvalid],  # early_stoppingの評価用データ
                    num_boost_round=10000,  # 最大学習サイクル数。early_stopping使用時は大きな値を入力
                    early_stopping_rounds=10,
                    verbose_eval=verbose_eval
                    )

    # スコア(RMSE)算出
    y_pred = gbm.predict(X[test])
    score = mean_squared_error(y_true=y[test], y_pred=y_pred, squared=False)
    scores.append(score)
print(f'RMSE={scores} \nRMSE mean={np.mean(scores)}')

# %%
