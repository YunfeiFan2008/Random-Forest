import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'n_estimators': [200, 300, 500, 800],
    'max_depth': [10, 15, 20, 30, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['auto', 'sqrt', 0.8],
    'bootstrap': [True, False]
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

tree = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)

search = RandomizedSearchCV(
    estimator=tree,
    param_distributions=param_grid,
    n_iter=50,
    cv=5,
    random_state=42,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
search.fit(X_train, y_train)
print(search.best_params_)