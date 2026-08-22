import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from main import run_model




disease_row = pd.DataFrame([{ #Dieased  person
    'age': 64, 'sex': 1, 'cp': 4, 'trestbps': 141, 'chol': 302,
     'restecg': 1, 'thalach': 98, 'exang': 1,
    'oldpeak': 5.2, 'slope': 3, 'ca': 1, 'thal': 6
}])

healthy_row = pd.DataFrame([{  #Healthy person
    'age': 46, 'sex': 1, 'cp': 3, 'trestbps': 101, 'chol': 211,
     'restecg': 0, 'thalach': 193, 'exang': 0,
    'oldpeak': 0.1, 'slope': 1, 'ca': 0, 'thal': 3
}])

new_patient = pd.DataFrame([{ #New patient
    'age': 50, 'sex': 1, 'cp': 2, 'trestbps': 120, 'chol': 250,
     'restecg': 0, 'thalach': 180, 'exang': 0,
    'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 2
}])


for row in [disease_row, healthy_row, new_patient]:
    print(run_model(row))