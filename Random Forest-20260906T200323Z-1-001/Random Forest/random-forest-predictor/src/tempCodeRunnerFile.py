import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

best_params = {
    'n_estimators': 300,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'max_depth': 20,
    'bootstrap': True,
    'random_state': 42  # always good to set for reproducibility
}
base_dir = os.path.dirname(__file__)
path = os.path.join(base_dir, 'data', 'housing.csv')
df = pd.read_csv(path)

wage = df['Growth in Wage (%)'].values  

inf = df['Inflation Rate (%)'].values

unemp = df['Employment Levels (000s)'].values 

ir = df['Interest Rate (%)'].values

hci = df['Housing Cost Index'].values

gdpG = df['GDP Growth Rate (%)'].values

print(len(wage), len(unemp), len(inf), len(hci))

combined_df = pd.DataFrame({
    'wageG': wage,
    'Unemployment Rate': unemp,
    'Inflation': inf,
    'Interest Rate': ir,
    'Housing Cost Index' : hci,
    'GDP Growth Rate': gdpG
})


X = combined_df[["wageG", "Unemployment Rate", "Interest Rate", "Housing Cost Index", "GDP Growth Rate"]].values
y = combined_df["Inflation"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(**best_params)

# Fit the model
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f"Test set Mean Squared Error: {mse:.4f}")