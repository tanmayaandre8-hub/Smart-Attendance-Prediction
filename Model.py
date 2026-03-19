import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, accuracy_score
import pickle

df = pd.read_csv("attendance_data.csv")

# Encoding
le_subject = LabelEncoder()
le_day = LabelEncoder()
le_time = LabelEncoder()

df["subject"] = le_subject.fit_transform(df["subject"])
df["day"] = le_day.fit_transform(df["day"])
df["time_slot"] = le_time.fit_transform(df["time_slot"])

# -------- REGRESSION --------
X = df[["subject", "day", "time_slot", "difficulty", "past_attendance"]]
y = df["attendance_percentage"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Linear Regression
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_pred = lin_model.predict(X_test)
print("Linear MAE:", mean_absolute_error(y_test, lin_pred))

# Random Forest
rf_model = RandomForestRegressor(n_estimators=50)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("Random Forest MAE:", mean_absolute_error(y_test, rf_pred))

# -------- CLASSIFICATION --------
y_class = df["high_attendance"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2)

log_model = LogisticRegression()
log_model.fit(X_train_c, y_train_c)
log_pred = log_model.predict(X_test_c)
print("Logistic Accuracy:", accuracy_score(y_test_c, log_pred))

# SAVE MODELS
pickle.dump(lin_model, open("linear.pkl", "wb"))
pickle.dump(rf_model, open("rf.pkl", "wb"))
pickle.dump(log_model, open("logistic.pkl", "wb"))

pickle.dump(le_subject, open("le_subject.pkl", "wb"))
pickle.dump(le_day, open("le_day.pkl", "wb"))
pickle.dump(le_time, open("le_time.pkl", "wb"))

print("All models saved!")