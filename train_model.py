# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestRegressor  # or LinearRegression
# import joblib

# # Load the dataset
# df = pd.read_csv("data/essay_llama3_8B_groq.csv")

# # Optional: Use only first 5000 rows for faster training
# df = df.head(5000)

# # Features and labels
# X_text = df["cleaned_text"]
# y = df["score"]

# # Vectorize
# vectorizer = TfidfVectorizer(max_features=5000)
# X = vectorizer.fit_transform(X_text)

# # Train the model
# model = RandomForestRegressor()
# model.fit(X, y)

# # Save the model and vectorizer
# joblib.dump(model, "model.pkl")
# joblib.dump(vectorizer, "vectorizer.pkl")
# print("✅ Model and vectorizer saved successfully.")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib

# Load the dataset
df = pd.read_csv("data/essay_llama3_8B_groq.csv").head(5000)

X_text = df["cleaned_text"]
y = df["score"]

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(X_text)

# Split into train/test for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("📊 Evaluation Results:")
print(f"✅ R² Score: {r2:.2f}")
print(f"📉 Mean Squared Error: {mse:.2f}")
print(f"📉 Mean Absolute Error: {mae:.2f}")

# Save the model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Model and vectorizer saved.")


