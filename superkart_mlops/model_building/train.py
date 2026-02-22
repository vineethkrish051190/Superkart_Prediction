# for data manipulation
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
# for model training, tuning, and evaluation
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError

api = HfApi()

Xtrain_path = "hf://datasets/vnsonly05/Superkart-Prediction/Xtrain.csv"
Xtest_path = "hf://datasets/vnsonly05/Superkart-Prediction/Xtest.csv"
ytrain_path = "hf://datasets/vnsonly05/Superkart-Prediction/ytrain.csv"
ytest_path = "hf://datasets/vnsonly05/Superkart-Prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)

# Encoding the categorical columns uning one hot encoding
cols_to_encode = ['Product_Sugar_Content','Product_Type','Store_Size','Store_Location_City_Type','Store_Type']

# Setup the transformer: 'remainder=passthrough' keeps the non-encoded columns
ct = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), cols_to_encode),
    remainder='passthrough'
)


# Define XGBoost Regressor model
xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

# creating pipeline
model_pipeline = make_pipeline(ct, xgb_model)

# Define hyperparameter grid
param_grid = {
    'xgbregressor__n_estimators': [50, 75, 100],
    'xgbregressor__max_depth': [2, 3, 4],
    'xgbregressor__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbregressor__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbregressor__learning_rate': [0.01, 0.05, 0.1],
    'xgbregressor__reg_lambda': [0.4, 0.5, 0.6],
}

#Grid Search CV
# cv=5 means 5-fold cross-validation (makes the model "reliable")
grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_absolute_error', # Evaluation metric
    verbose=1,
    n_jobs=-1 # Use all CPU cores for speed
)

# train the search
grid_search.fit(Xtrain, ytrain)

# Best model
best_model = grid_search.best_estimator_
print("Best Params:\n", grid_search.best_params_)

# Predict on training set
y_pred_train = best_model.predict(Xtrain)

# Predict on test set
y_pred_test = best_model.predict(Xtest)

# Evaluation
def get_metrics(model, X, y_true, label):
    y_pred = model.predict(X)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n[{label} Metrics]")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")
    return mae

print("\n--- Final Model Evaluation ---")
test_mae = get_metrics(best_model, Xtest, ytest, "Best Model Test Set")

# Save best model
joblib.dump(best_model, "superkart_best_model.joblib")

# Upload to Hugging Face
repo_id = "vnsonly05/Superkart-Prediction"
repo_type = "model"

api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Model Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Model Space '{repo_id}' created.")

# create_repo("superkart_best_model", repo_type="model", private=False)
api.upload_file(
    path_or_fileobj="superkart_best_model.joblib",
    path_in_repo="superkart_best_model.joblib",
    repo_id=repo_id,
    repo_type=repo_type,
)
