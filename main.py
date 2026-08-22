import pandas as pd
import numpy as np
from pandas.io.formats.printing import PrettyDict
from sklearn.metrics._plot.regression import PredictionErrorDisplay
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
# 1. Load Data
names = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']
df = pd.read_csv("data/processed.cleveland.data", names=names)

# 2. Process Target and Missing Values
df['target'] = (df['num'] > 0).astype(int)
df = df.replace('?', np.nan)
df['ca'] = pd.to_numeric(df['ca'])
df['thal'] = pd.to_numeric(df['thal'])

# Fill missing values with median so XGBoost doesn't stumble on NaNs
df['ca'] = df['ca'].fillna(df['ca'].median())
df['thal'] = df['thal'].fillna(df['thal'].median())

X = df.drop(['target','num','fbs'], axis=1)
y = df['target']

# 3. Fixed Hyperparameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 4,                  # Set to a stable 4
    'eta': 0.03,                     # Slower learning rate to prevent overfitting
    'subsample': 0.8,                # FIX: Train on 80% of rows (not 3%!)
    'colsample_bytree': 0.8          # Train on 80% of columns
}

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
y_train_clean = y_train.astype(int)
model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train_clean)
def run_model(predictions):
    return model.predict(predictions)


# Dtrain = xgb.DMatrix(X_train, label=y_train_clean)
# model_test = xgb.cv(
#     params,
#     Dtrain,
#     num_boost_round=100,             # Increased to give the 0.03 eta time to learn
#     nfold=5,
#     stratified=True,
#     early_stopping_rounds=15,
#     verbose_eval=False
# )

# print(model_test.tail())
