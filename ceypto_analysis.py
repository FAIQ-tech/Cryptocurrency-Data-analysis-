# ================================================================
#  Bitcoin Price Prediction Using Linear Regression
#  Author: Muhammad Faiq Hayat
# ================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

# --- Load Dataset ---
df = pd.read_csv('crypto_data_updated_13_november.csv')

# --- Feature Engineering ---
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df.drop('Date', axis=1, inplace=True)

# --- Define Features and Target ---
X = df.drop('Close (BTC)', axis=1)
y = df['Close (BTC)']

# --- Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Create & Train Model ---
model = make_pipeline(StandardScaler(), LinearRegression())
model.fit(X_train, y_train)

# --- Evaluate Model ---
y_pred = model.predict(X_test)
r2_train = r2_score(y_train, model.predict(X_train))
r2_test = r2_score(y_test, y_pred)

print(f"Training R² Score: {r2_train:.3f}")
print(f"Testing R² Score: {r2_test:.3f}")
