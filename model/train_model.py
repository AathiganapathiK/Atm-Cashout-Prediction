import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np
import joblib
import pickle


df=pd.read_csv('data/christ college atm.csv')
     
df.columns = df.columns.str.strip()  # Clean column names


# Convert 'Transaction Date' to datetime
df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], format="%d-%m-%Y")

# Add `is_festival` column
df["is_festival"] = df["Festival Religion"].apply(lambda x: 1 if str(x).strip() != "NH" else 0)


# Encode weekday as numerical
weekday_map = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6, "MONDAY":0, "TUESDAY":1, "WEDNESDAY":2, "THURSDAY":3,
    "FRIDAY":4, "SATURDAY":5, "SUNDAY":6
}
df["Day_of_Week"] = df["Weekday"].map(weekday_map)


# Sort by date
df.sort_values("Transaction Date", inplace=True)
df.head()

# Create the target column: next day's 'Total amount Withdrawn'
df["Next_Day_Withdrawn"] = df["Total amount Withdrawn"].shift(-1)

# Drop rows with missing target (i.e., last row)
df.dropna(subset=["Next_Day_Withdrawn"], inplace=True)


# Step 1: Select features (X) and target (y)
features = [
    "No Of Withdrawals",
    "is_festival",
    "Day_of_Week"
]

X = df[features]

# Compute the median safely
median_withdrawals = X["No Of Withdrawals"].median()

# Fill missing values safely
X.fillna({
    "No Of Withdrawals": median_withdrawals,
    "is_festival": 0
}, inplace=True)

# Forward fill for day of week
X["Day_of_Week"] = X["Day_of_Week"].ffill()


y = df["Next_Day_Withdrawn"]


# Train-test split again
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train the model again
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set and calculate MAE
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# Predict for the next day
next_day_features = X.iloc[[-1]]
predicted_withdrawal = model.predict(next_day_features)[0]

# Determine buffer
is_festival = int(next_day_features["is_festival"].values[0])
day_of_week = int(next_day_features["Day_of_Week"].values[0])
buffer_percent = 0.15 if is_festival or day_of_week in [5, 6] else 0.10


# Final amount to load
cash_to_load = predicted_withdrawal * (1 + buffer_percent)

# Determine the next day's date and name
last_date = df["Transaction Date"].max()
next_day = last_date + pd.Timedelta(days=1)
next_day_name = next_day.strftime("%A")


# Final output
predicted_withdrawal, buffer_percent, cash_to_load, next_day.strftime("%Y-%m-%d"), next_day_name

# Save the model properly
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)
