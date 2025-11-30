This folder contains the various intermediate output files within the ML pipeline as described below:
* cat_features_XXXX: Raw data processed into features of our choosing for both models, initially intended only for CatBoost. XGBoost adapts the categorical data by employing One-Hot Encoding.
* catboost_model_log_flow: File containing the model parameters for CatBoost. Predicts log(1+fuel_flow_rate), hence the name log_flow
* xgb_linear_ohe: The encoding file that carries over the relevant information when applying the model on rank and final datasets.
* xgb_model_linear: Zip file containing a json file with all the model parameters for running XGBoost. Predicts fuel_kg, hence the name linear
