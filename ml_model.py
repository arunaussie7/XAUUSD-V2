import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("xauusd_M5.csv")  # Changed from H5 to M5

# Define the target (1 = Bullish, 0 = Bearish)
df['target'] = np.where(df['close'] > df['open'], 1, 0)

# Select features
features = ['open', 'high', 'low', 'close', 'tick_volume']
X = df[features]
y = df['target']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2f}")

# Save the model for later use
import joblib
joblib.dump(model, "ml_model.pkl") 