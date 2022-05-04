# lightgbm_early_stopping
 LightGBMでearly_stoppingを使用する際のサンプルコード

|クロスバリデーション|API|旧指定方法<br>('early_stopping_rounds'<br>引数)|現在の推奨方法<br>(コールバック関数<br>'early_stopping()')|
|---|---|---|---|
|**なし**|**Training API**|[lightgbm_training_api_old.py](https://github.com/c60evaporator/lightgbm_early_stopping/blob/main/lightgbm_training_api_old.py)|[lightgbm_training_api_new.py](https://github.com/c60evaporator/lightgbm_early_stopping/blob/main/lightgbm_training_api_new.py)|
|**なし**|**Scikit-Learn API**|[lightgbm_sklearn_api_old.py](https://github.com/c60evaporator/lightgbm_early_stopping/blob/main/lightgbm_sklearn_api_old.py)|[lightgbm_sklearn_api_new.py](https://github.com/c60evaporator/lightgbm_early_stopping/blob/main/lightgbm_sklearn_api_new.py)|
|**あり**|**Training API**<br>(スクラッチ実装)|[lightgbm_training_api_cv_scratch_old.py](https://github.com/c60evaporator/lightgbm_early_stopping/blob/main/lightgbm_training_api_cv_scratch_old.py)|[lightgbm_training_api_cv_scratch_new.py](https://github.com/c60evaporator/lightgbm_early_stopping/blob/main/lightgbm_training_api_cv_scratch_new.py)|
|**あり**|**Training API**<br>(`lightgbm.cv`メソッド)|[lightgbm_training_api_cv_lgbcv_old.py](https://github.com/c60evaporator/lightgbm_early_stopping/blob/main/lightgbm_training_api_cv_lgbcv_old.py)|**[lightgbm_training_api_cv_lgbcv_new.py](https://github.com/c60evaporator/lightgbm_early_stopping/blob/main/lightgbm_training_api_cv_lgbcv_new.py)**|
|**あり**|**Scikit-Learn API**<br>(スクラッチ実装)|[lightgbm_sklearn_api_cv_old.py](https://github.com/c60evaporator/lightgbm_early_stopping/blob/main/lightgbm_sklearn_api_cv_old.py)|[lightgbm_sklearn_api_cv_new.py](https://github.com/c60evaporator/lightgbm_early_stopping/blob/main/lightgbm_sklearn_api_cv_new.py)|
|**あり**|**Scikit-Learn API**<br>(`cross_val_score`メソッド)|-|**[lightgbm_sklearn_api_cv_crossvalscore.py](https://github.com/c60evaporator/lightgbm_early_stopping/blob/main/lightgbm_sklearn_api_cv_crossvalscore.py)**|
