import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
data = pd.read_csv(r"src\data\housing.csv")
data2 = pd.read_csv(r"src\data\economic_data_2020_2023.csv")

X = data[['Housing Cost Index',
          'Interest Rate (%)',
          'Inflation Rate (%)',
          'Growth in Wage (%)',
          'GDP Growth Rate (%)']].values
y = data['Employment Levels (000s)'].values

X_test = data2[['Housing Cost Index',
            'Interest Rate (%)',
            'Inflation Rate (%)',
            'Growth in Wage (%)',
            'GDP Growth Rate (%)']].values
y_test_actual = data2['Employment Levels (000s)'].values

class DecisionTree:
    def __init__(self, max_depth=None): #called when DecisionTree() is called
        self.max_depth = max_depth 
        # the max depth of the tree shows how tall the "tree is allowed to grow
        self.min_samples_split = 3
        self.tree = None
    def fit(self, X, y): #recursively builds the decision tree
        self.tree = self.build_tree(X, y)
    def predict(self, X): 
        predictions = [self.predict_single(row, self.tree) for row in X]
        return np.array(predictions)
    def build_tree(self, X, y, depth=0): 
        if len(y) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            return {"leaf": np.mean(y)}
        
        node = self.best_split(X, y)
        if node["feature"] is None:
            return {"leaf": np.mean(y)}
        
        X_left, y_left, X_right, y_right = node["groups"]
        del node["groups"] 
        
        if len(y_left) == 0 or len(y_right) == 0:
            return {"leaf": np.mean(y)}
        
        node["left"] = self.build_tree(X_left, y_left, depth + 1)
        node["right"] = self.build_tree(X_right, y_right, depth + 1)
    
        return node

    def best_split(self, X, y):
        best_index, best_value, best_mse = None, None, float("inf")
        n_features = int(np.sqrt(X.shape[1]))
        features = []
        while len(features) < n_features:
            feature = np.random.randint(0, X.shape[1])
            if feature not in features:
                features.append(feature)
        for index in features:
            for row in X:
                threshold = row[index]
                groups = self.split(X, y, index, threshold)
                mse = self.mse_index(groups) 
            
                if mse <= best_mse:  
                    best_index = index
                    best_value = threshold
                    best_mse = mse
                    best_groups = groups
    
        return {"feature": best_index, "threshold": best_value, "groups": best_groups} 
        
    def split(self, X, y, feature_idx, threshold):
        column_values = X[:, feature_idx]
        belowmask = column_values <= threshold
        abovemask = column_values > threshold
        
        X_below = X[belowmask]
        y_below = y[belowmask]
        X_above = X[abovemask]
        y_above = y[abovemask]
        
        return X_below, y_below, X_above, y_above

    def predict_single(self, x, tree):
        if "leaf" in tree:
            return tree["leaf"]
    
        feature_idx = tree["feature"]
        threshold = tree["threshold"]

        if x[feature_idx] <= threshold:
            return self.predict_single(x, tree["left"])
        else:
            return self.predict_single(x, tree["right"])
        
    def calculate_MSE(self, y):
        n = len(y)
        if n == 0:
            return 0
        mean = np.mean(y)
        mse = np.sum( (y - mean) ** 2)/n
        return mse
    def mse_index(self, groups):
        X_below, y_below, X_above, y_above = groups
        n_total = len(y_below) + len(y_above)
        mse_below = self.calculate_MSE(y_below)
        mse_above = self.calculate_MSE(y_above)
        weighted_mse = (len(y_below)/n_total) * mse_below + (len(y_above)/n_total) * mse_above
        
        return weighted_mse


class RandomForest:
    def __init__(self, n_trees=50, max_depth=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
    def bootstrapping(self, X, y):
        bootstrap_indices = np.random.randint(low=0, high=len(X), size=len(X))
        return X[bootstrap_indices], y[bootstrap_indices]
    def fit(self, X, y):
        self.trees = []
        for i in range(self.n_trees):
            X_boot, y_boot = self.bootstrapping(X, y)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
            print(f"Building trees... {i+1}/{self.n_trees}")
    def predict(self, X):
        all_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(all_preds, axis=0)  # average across trees

forest = RandomForest(n_trees=100, max_depth=100)
forest.fit(X, y)
predictions = forest.predict(X_test)
for date, pred in zip(data2["Date"], predictions):
    print(f"Date: {date}, Predicted: {pred:.2f}")
r2 = r2_score(y_test_actual, predictions)
print(f"R2 Score: {r2:.4f}")

indices = list(range(0, len(data2), 12))
plt.figure(figsize=(12,6))
plt.plot(data2['Date'], y_test_actual, label='Actual (Monthly)', marker='o')
plt.plot(data2['Date'], predictions, label='Predicted (Monthly)', marker='x')
plt.xlabel("Date")
plt.ylabel("Employment Levels (000s)")
plt.title("Actual vs Predicted Employment Levels")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()