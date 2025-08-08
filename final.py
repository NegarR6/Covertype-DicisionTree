

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import networkx as nx

plt.rcParams.update({'figure.max_open_warning': 0})

print(" LOADING DATASET ")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
columns = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'
] + [f'Wilderness_Area_{i}' for i in range(4)] + [f'Soil_Type_{i}' for i in range(40)] + ['Cover_Type']

data = pd.read_csv(url, names=columns, compression='gzip')
print(data.info())
print("\nMissing values per column:\n", data.isnull().sum())

X = data.drop('Cover_Type', axis=1)
y = data['Cover_Type']

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

# Save splits to CSV 
X_train.to_csv("X_train.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
X_val.to_csv("X_val.csv", index=False)
y_val.to_csv("y_val.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print(f"\nTrain size: {len(X_train)}")
print(f"Validation size: {len(X_val)}")
print(f"Test size: {len(X_test)}")

continuous_features = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'
]

def quartile_binning(series, q_dict=None):
    if q_dict is None:
        Q1 = series.quantile(0.25)
        Q2 = series.quantile(0.50)
        Q3 = series.quantile(0.75)
    else:
        Q1, Q2, Q3 = q_dict
    bins = [-float('inf'), Q1, Q2, Q3, float('inf')]
    labels = ['Q1', 'Q2', 'Q3', 'Q4']
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True), (Q1, Q2, Q3)

quartiles = {}
X_train_q = pd.DataFrame()
X_val_q = pd.DataFrame()
X_test_q = pd.DataFrame()

for feature in continuous_features:
    X_train_q[feature + '_Q'], quartiles[feature] = quartile_binning(X_train[feature])
    X_val_q[feature + '_Q'], _ = quartile_binning(X_val[feature], quartiles[feature])
    X_test_q[feature + '_Q'], _ = quartile_binning(X_test[feature], quartiles[feature])

X_train_d = X_train_q.copy()
X_val_d = X_val_q.copy()
X_test_d = X_test_q.copy()

# Quartile distribution plots 
plt.figure(figsize=(12, 8))
for i, feature in enumerate(continuous_features[:4], 1):
    plt.subplot(2, 2, i)
    sns.countplot(x=feature + '_Q', data=X_train_d)
    plt.title(f'Distribution of {feature} (Quartile)')
plt.tight_layout()
plt.show()


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, criterion='entropy'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = None
        self.feature_names = None

    def entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs + 1e-10))

    def gini_index(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)

    def information_gain(self, y, y_left, y_right):
        parent_score = self.entropy(y) if self.criterion == 'entropy' else self.gini_index(y)
        weight_left = len(y_left) / len(y)
        weight_right = len(y_right) / len(y)
        child_score = weight_left * (self.entropy(y_left) if self.criterion == 'entropy' else self.gini_index(y_left)) + \
                      weight_right * (self.entropy(y_right) if self.criterion == 'entropy' else self.gini_index(y_right))
        return parent_score - child_score

    def best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_value = None
        best_gini = 1.0
        for feature in range(X.shape[1]):
            for value in np.unique(X.iloc[:, feature]):
                y_left = y[X.iloc[:, feature] == value]
                y_right = y[X.iloc[:, feature] != value]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                gain = self.information_gain(y, y_left, y_right)
                gini = self.gini_index(y)
                if gain > best_gain and len(y_left) >= self.min_samples_split and len(y_right) >= self.min_samples_split:
                    best_gain = gain
                    best_feature = feature
                    best_value = value
                    best_gini = gini
        return best_feature, best_value, best_gain, best_gini

    def build_tree(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or (self.max_depth and depth >= self.max_depth) or len(y) < self.min_samples_split:
            return {'leaf': True, 'value': Counter(y).most_common(1)[0][0]}
        best_feature, best_value, gain, gini = self.best_split(X, y)
        if best_feature is None:
            return {'leaf': True, 'value': Counter(y).most_common(1)[0][0]}
        left_mask = X.iloc[:, best_feature] == best_value
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[~left_mask], y[~left_mask]
        return {
            'leaf': False, 'feature': best_feature, 'value': best_value,
            'gain': gain, 'gini': gini,
            'left': self.build_tree(X_left, y_left, depth + 1),
            'right': self.build_tree(X_right, y_right, depth + 1)
        }

    def fit(self, X, y):
        self.feature_names = X.columns
        self.tree = self.build_tree(X, y)

    def predict_row(self, row, node):
        if node.get('leaf', False):
            return node['value']
        if row.iloc[node['feature']] == node['value']:
            return self.predict_row(row, node['left'])
        else:
            return self.predict_row(row, node['right'])

    def predict(self, X):
        return np.array([self.predict_row(row, self.tree) for _, row in X.iterrows()])

    def prune(self, X_val, y_val, node):
        if node.get('leaf', False):
            return node
        node['left'] = self.prune(X_val, y_val, node['left'])
        node['right'] = self.prune(X_val, y_val, node['right'])
        preds_with = self.predict(X_val)
        acc_with = accuracy_score(y_val, preds_with)
        majority_class = Counter(y_val).most_common(1)[0][0]
        acc_without = accuracy_score(y_val, [majority_class]*len(y_val))
        if acc_without >= acc_with:
            return {'leaf': True, 'value': majority_class}
        return node


# Train

dt = DecisionTree(max_depth=5, min_samples_split=10, criterion='entropy')
dt.fit(X_train_d, y_train)

def print_tree(node, depth=0, feature_names=None):
    if node.get('leaf', False):
        print("  " * depth + f"Leaf: {node['value']}")
        return
    feature_name = feature_names[node['feature']]
    print("  " * depth + f"{feature_name} = {node['value']} (Gain: {node['gain']:.4f}, Gini: {node['gini']:.4f})")
    print_tree(node['left'], depth + 1, feature_names)
    print("  " * depth + f"{feature_name} != {node['value']} (Gain: {node['gain']:.4f}, Gini: {node['gini']:.4f})")
    print_tree(node['right'], depth + 1, feature_names)

print("\n Decision Tree Structure ")
print_tree(dt.tree, feature_names=dt.feature_names)

# Accuracy before pruning
print("\n Accuracy BEFORE Pruning ")
for name, X_set, y_set in [("Train", X_train_d, y_train), ("Validation", X_val_d, y_val), ("Test", X_test_d, y_test)]:
    preds = dt.predict(X_set)
    print(f"{name}: {accuracy_score(y_set, preds):.4f}")

# Apply pruning
dt.tree = dt.prune(X_val_d, y_val, dt.tree)

# Accuracy after pruning
print("\n Accuracy AFTER Pruning ")
for name, X_set, y_set in [("Train", X_train_d, y_train), ("Validation", X_val_d, y_val), ("Test", X_test_d, y_test)]:
    preds = dt.predict(X_set)
    print(f"{name}: {accuracy_score(y_set, preds):.4f}")

preds_test = dt.predict(X_test_d)

# Confusion Matrix
cm = confusion_matrix(y_test, preds_test)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(1,8), yticklabels=range(1,8))
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature Importance
importance = defaultdict(int)
def compute_importance(node):
    if node.get('leaf', False):
        return
    importance[dt.feature_names[node['feature']]] += 1
    compute_importance(node['left'])
    compute_importance(node['right'])
compute_importance(dt.tree)

imp_series = pd.Series(importance).sort_values(ascending=False)
if imp_series.empty:
    print("No split-based feature importance found (empty tree?)")
else:
    df_imp = imp_series.reset_index()
    df_imp.columns = ['feature', 'count']
    plt.figure(figsize=(10,6))

    ax = sns.barplot(data=df_imp, x='count', y='feature', palette='viridis', hue=df_imp['feature'], dodge=False)
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    plt.title('Feature Importance (Split Counts)')
    plt.xlabel('Count of Splits')
    plt.ylabel('Feature')
    plt.show()


class_labels = list(range(1, 8))  # enforce full class set 1..7
accuracies = []
class_counts = []
for cls in class_labels:
    idx = (y_test == cls)
    n = int(idx.sum())
    class_counts.append(n)
    if n == 0:
        acc = 0.0
    else:
        acc = float(accuracy_score(y_test[idx], preds_test[idx]))
    accuracies.append(acc)

# Print table for clarity
print("\nPer-class accuracy (classes 1..7):")
for cl, acc, cnt in zip(class_labels, accuracies, class_counts):
    print(f" Class {cl}: accuracy = {acc:.4f}     (samples in test: {cnt})")

df_cls = pd.DataFrame({'class': [str(c) for c in class_labels], 'accuracy': accuracies})
plt.figure(figsize=(8,6))
ax = sns.barplot(data=df_cls, x='class', y='accuracy', palette='coolwarm', hue=df_cls['class'], dodge=False)
if ax.get_legend() is not None:
    ax.get_legend().remove()
plt.title('Per-Class Accuracy (All classes)')
plt.ylim(0,1)
# Annotate values on bars
for i, a in enumerate(accuracies):
    ax.text(i, a + 0.02, f"{a:.2f}", ha='center')
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.show()

# Tree Graph

def add_nodes_edges(tree, parent=None, graph=None, node_id=0):
    if graph is None:
        graph = nx.DiGraph()
    if tree.get('leaf', False):
        label = f"Leaf: {tree['value']}"
    else:
        label = f"{dt.feature_names[tree['feature']]} = {tree['value']}\nGain: {tree['gain']:.3f}, Gini: {tree['gini']:.3f}"
    graph.add_node(node_id, label=label)
    if parent is not None:
        graph.add_edge(parent, node_id)
    if not tree.get('leaf', False):
        node_id = add_nodes_edges(tree['left'], node_id, graph, node_id+1)
        node_id = add_nodes_edges(tree['right'], node_id, graph, node_id+1)
    return node_id + 1 if tree.get('leaf', False) else node_id

def plot_tree_graph(tree):
    G = nx.DiGraph()
    add_nodes_edges(tree, graph=G)
    pos = nx.spring_layout(G, seed=42)
    labels = nx.get_node_attributes(G, 'label')
    plt.figure(figsize=(14,10))
    nx.draw(G, pos, with_labels=False, node_size=2000, node_color="lightblue", arrows=True)
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    plt.title("Decision Tree Visualization")
    plt.show()

plot_tree_graph(dt.tree)
