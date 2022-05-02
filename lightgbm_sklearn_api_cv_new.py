# %% Scikit-Learn API + 新指定方法(コールバック関数early_stopping) + cross_val_score_eval_setメソッドでクロスバリデーション
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
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

# クロスバリデーションのデータ分割
scores=[]
for i, (train, test) in enumerate(cv.split(X, y)):
    ###### ここからがLightGBMの実装 ######
    # 使用するパラメータ
    param = {'objective': 'regression',  # 最小化させるべき損失関数
            'random_state': 42,  # 乱数シード
            'boosting_type': 'gbdt',  # boosting_type
            'n_estimators': 10000  # 最大学習サイクル数。early_stopping使用時は大きな値を入力
            }
    verbose_eval = 0  # この数字を1にすると学習時のスコア推移がコマンドライン表示される
    # early_stoppingを指定してLightGBM学習
    lgbr = lgb.LGBMRegressor(**param)
    lgbr.fit(X[train], y[train], 
            eval_metric='rmse',  # early_stoppingの評価指標(学習用の'metric'パラメータにも同じ指標が自動入力される)
            eval_set=[(X[test], y[test])],
            callbacks=[lgb.early_stopping(stopping_rounds=10, 
                                verbose=True), # early_stopping用コールバック関数
                           lgb.log_evaluation(verbose_eval)] # コマンドライン出力用コールバック関数
            )

    # スコア(RMSE)算出
    y_pred = lgbr.predict(X[test])

    score = mean_squared_error(y_true=y[test], y_pred=y_pred, squared=False)
    scores.append(score)
print(f'RMSE={scores} \nRMSE mean={np.mean(scores)}')

# %%
