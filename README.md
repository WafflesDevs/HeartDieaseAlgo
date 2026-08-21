# Machine Learning Model for Heart Disease dataset (UCI Irivines)

A Machine Learning Model using XGBoosts model, Fined tuned after a few hours and modified data.

## 🔗 Links

[Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
[XGBoost](https://xgboost.readthedocs.io/en/stable/)

## How to use it

To make a "patient" declare a dataframe from pandas and call the model to predict it.

```python
new_patient = pd.DataFrame([{ #New patient
    'age': 50, 'sex': 1, 'cp': 2, 'trestbps': 120, 'chol': 250,
    'fbs': 0, 'restecg': 0, 'thalach': 180, 'exang': 0,
    'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 2
}])
```

This is how you call the prediction from the model

```python
model.predict(new_patient)
```

The model returns the answer in this format  (Either 1 or 0)

```python
[1] # Patient has Heart Diease
[0] # Patient does not have it
```

