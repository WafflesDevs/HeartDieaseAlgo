import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
df = pd.read_csv('data/processed.cleveland.data', names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'])
selector = VarianceThreshold(threshold=0.001)
num_df = df.drop(columns=["ca", "thal"])
sel = VarianceThreshold(threshold=0.001)
# Fit the selector to normalized head_df
sel.fit(num_df / num_df.mean())

# Create a boolean mask 
mask = sel.get_support()

# Apply the mask to create a reduced DataFrame
reduced_df = num_df.loc[:, mask]

print(f"Dimensionality reduced from {df.shape[1]} to {reduced_df.shape[1]}.")
print(reduced_df.info())