import numpy as np
import pandas as pd
from keras.losses import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

df1 = pd.read_pickle("../workflow_files/list_files/price_attached_vectors.pkl")
df2 = pd.read_pickle("../workflow_files/list_files/price_attached_vectors_test.pkl")
df_concat = pd.concat([df1, df2], ignore_index=True)

# Extract rows 30 to 50 (inclusive of 30, exclusive of 50)
df_test = df_concat.iloc[890:1050].copy()
# Extract remaining rows (excluding 30 to 50)
df = pd.concat([df_concat.iloc[:890], df_concat.iloc[1050:]]).copy()
# Reset index (optional)
df_test.reset_index(drop=True, inplace=True)
df.reset_index(drop=True, inplace=True)

# Training features from df
train_features = np.stack(df['fingpt_vectors'].tolist())  # Shape: (n_samples, 768)
train_target = df['price'].values  # Shape: (n_samples,)

# Test features from df_test
test_features = np.stack(df_test['fingpt_vectors'].tolist())  # Shape: (n_samples, 768)
test_target = df_test['price'].values  # Shape: (n_samples,)

# 2. Apply PCA for dimensionality reduction (if needed)
pca = PCA(n_components=768)  # Reduce to 100 dimensions (you can adjust this number)
train_features_reduced = pca.fit_transform(train_features)  # Shape: (n_samples, 100)
test_features_reduced = pca.transform(test_features)  # Apply same transformation to test data

# 3. Train the Random Forest Regressor on the training data
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train_features_reduced, train_target)

# 4. Make predictions on the test data (unseen data)
predictions = model.predict(test_features_reduced)

# 5. Evaluate the model performance using MSE and R² score
mse = mean_squared_error(test_target, predictions)
r2 = r2_score(test_target, predictions)

# Print the evaluation metrics
print(f'MSE: {mse:.4f}')
print(f'R² Score: {r2:.4f}')
