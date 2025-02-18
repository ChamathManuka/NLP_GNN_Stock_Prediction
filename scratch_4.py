import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

df1 = pd.read_pickle("workflow_files/list_files/price_attached_vectors.pkl")
df2 = pd.read_pickle("workflow_files/list_files/price_attached_vectors_test.pkl")
df_concat = pd.concat([df1, df2], ignore_index=True)


# Extract rows 30 to 50 (inclusive of 30, exclusive of 50)
df_test = df_concat.iloc[890:1050].copy()
# Extract remaining rows (excluding 30 to 50)
df = pd.concat([df_concat.iloc[:890], df_concat.iloc[1050:]]).copy()
# Reset index (optional)
df_test.reset_index(drop=True, inplace=True)
df.reset_index(drop=True, inplace=True)

# Assuming you have the 'fingpt_vectors' as a list of vectors (numpy arrays) and 'price' as stock prices
fingpt_vectors = np.stack(df['fingpt_vectors'].values)  # Convert the list of vectors into a 2D numpy array
prices = df['price'].values  # Get the prices

pca = PCA(n_components=100)  # Reduce to 50 dimensions
fingpt_reduced = pca.fit_transform(np.stack(df['fingpt_vectors'].tolist()))
df['fingpt_vectors'] = list(fingpt_reduced)


# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score
#
# # Train a Random Forest model
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(fingpt_reduced, df['price'])
#
# # Predict on the same data (you can use a separate test set for evaluation)
# predictions = model.predict(fingpt_reduced)
#
# # Evaluate the model
# mse = mean_squared_error(df['price'], predictions)
# r2 = r2_score(df['price'], predictions)
#
# print(f'MSE: {mse:.4f}')
# print(f'R² Score: {r2:.4f}')

from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(fingpt_reduced, df['price'])

print("Best parameters:", grid_search.best_params_)
