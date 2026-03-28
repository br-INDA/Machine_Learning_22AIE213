from collections import Counter         
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA                       
from sklearn.tree import DecisionTreeClassifier, plot_tree  
from sklearn.model_selection import train_test_split        
from sklearn.inspection import DecisionBoundaryDisplay


X_full = np.load("/mnt/user-data/uploads/X_features.npy")   
y_full = np.load("/mnt/user-data/uploads/y_labels.npy")     

print(f"Loaded  X: {X_full.shape}  |  y: {y_full.shape}")
print(f"Classes : {np.unique(y_full)}  |  Counts: {Counter(y_full.tolist())}")

#Sampling
np.random.seed(42)                                           #reproducibility
sample_idx = np.random.choice(len(X_full), size=3000, replace=False)
X_sample   = X_full[sample_idx]    
y_sample   = y_full[sample_idx]    

#PCA dimensionality reduction 
pca_feat      = PCA(n_components=10, random_state=42)
X_pca10       = pca_feat.fit_transform(X_sample)   
FEATURE_NAMES = [f"pca_{i}" for i in range(10)]    

#Build a tidy DataFrame so tree functions can reference columns by name
df           = pd.DataFrame(X_pca10, columns=FEATURE_NAMES)
df["target"] = y_sample           
TARGET_COL   = "target"


#Entropy

def equal_width_binning(series: pd.Series, n_bins: int = 4) -> pd.Series:
    
    return pd.cut(series, bins=n_bins, labels=False)


def equal_frequency_binning(series: pd.Series, n_bins: int = 4) -> pd.Series:
    
    return pd.qcut(series, q=n_bins, labels=False, duplicates="drop")


def bin_column(series: pd.Series,
               binning_type: str = "width",
               n_bins: int = 4) -> pd.Series:
    if binning_type == "frequency":
        return equal_frequency_binning(series, n_bins)
    elif binning_type == "width":
        return equal_width_binning(series, n_bins)
    else:
        raise ValueError(f"Unknown binning_type='{binning_type}'. "
                         f"Choose 'width' or 'frequency'.")


def calculate_entropy(labels) -> float:
    
    labels = np.array(labels)
    n      = len(labels)
    if n == 0:
        return 0.0                           

    counts  = Counter(labels)               
    entropy = 0.0
    for count in counts.values():
        p_i = count / n                     
        if p_i > 0:                         
            entropy -= p_i * np.log2(p_i)  

    return entropy


entropy_val = calculate_entropy(y_sample)
print(f"\n[A1] Dataset Entropy : {entropy_val:.4f}")

#Gini Index
def calculate_gini(labels) -> float:
    
    labels = np.array(labels)
    n      = len(labels)
    if n == 0:
        return 0.0

    counts = Counter(labels)
    gini   = 1.0                  
    for count in counts.values():
        p_j   = count / n
        gini -= p_j ** 2          

    return gini


gini_val = calculate_gini(y_sample)
print(f"[A2] Dataset Gini Index : {gini_val:.4f}")


# IG
def information_gain(data: pd.DataFrame,
                     feature_col: str,
                     target_col: str,
                     binning_type: str = "width",
                     n_bins: int = 4) -> float:
    
    parent_entropy = calculate_entropy(data[target_col])

    n       = len(data)
    feature = data[feature_col].copy()

    if feature.dtype in [np.float64, np.float32, float]:
        feature = bin_column(feature, binning_type=binning_type, n_bins=n_bins)

    weighted_entropy = 0.0
    for val in feature.dropna().unique():       
        subset  = data[feature == val]          
        weight  = len(subset) / n              
        weighted_entropy += weight * calculate_entropy(subset[target_col])

    return parent_entropy - weighted_entropy


def find_root_node(data: pd.DataFrame,
                   feature_cols,
                   target_col: str):
    
    gains = {}
    for col in feature_cols:
        gains[col] = information_gain(data, col, target_col)
        print(f"   IG [{col:>8}]: {gains[col]:.4f}")

    best_feature = max(gains, key=gains.get)   
    return best_feature, gains


print("\n[A3] Information Gain per PCA feature:")
root_feat, ig_scores = find_root_node(df, FEATURE_NAMES, TARGET_COL)
print(f"\n   Best Root Node: '{root_feat}'  (IG = {ig_scores[root_feat]:.4f})")


#Decsion Tree

class DecisionTreeNode:
    
    def __init__(self):
        self.feature  = None    #splitting feature 
        self.children = {}      
        self.label    = None    #majority class prediction 


class MyDecisionTree:
    

    def __init__(self,
                 max_depth: int = 5,
                 min_samples_split: int = 5,
                 binning_type: str = "width",
                 n_bins: int = 4):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.binning_type      = binning_type
        self.n_bins            = n_bins
        self.root              = None   
        self._bin_maps         = {}     


    def _bin(self, series: pd.Series, col: str) -> pd.Series:
        
        if col not in self._bin_maps:
            if self.binning_type == "frequency":
                _, edges = pd.qcut(series, q=self.n_bins, retbins=True,
                                   labels=False, duplicates="drop")
            else:
                _, edges = pd.cut(series, bins=self.n_bins, retbins=True,
                                  labels=False)
            self._bin_maps[col] = edges   

        return pd.cut(series, bins=self._bin_maps[col],
                      labels=False, include_lowest=True)

    def _preprocess(self, data: pd.DataFrame, cols: list) -> pd.DataFrame:
        
        d = data.copy()
        for c in cols:
            if d[c].dtype in [np.float64, np.float32, float]:
                d[c] = self._bin(d[c], c)
        return d

    def _build(self,
               data: pd.DataFrame,
               feature_cols: list,
               target_col: str,
               depth: int) -> DecisionTreeNode:
       
        node   = DecisionTreeNode()
        labels = data[target_col]

        if (len(labels.unique()) == 1           
                or depth >= self.max_depth       
                or len(data) < self.min_samples_split  
                or not feature_cols):            
            node.label = labels.mode()[0]        
            return node

        gains = {c: information_gain(data, c, target_col) for c in feature_cols}
        best  = max(gains, key=gains.get)

        if gains[best] <= 0:
            node.label = labels.mode()[0]
            return node

        node.feature = best
        rest = [c for c in feature_cols if c != best]

        for val in data[best].dropna().unique():
            subset = data[data[best] == val]         
            node.children[val] = (
                self._build(subset, rest, target_col, depth + 1)  
                if len(subset) > 0
                else self._leaf(labels)   
            )

        return node

    def _leaf(self, labels: pd.Series) -> DecisionTreeNode:
        
        n       = DecisionTreeNode()
        n.label = labels.mode()[0]   
        return n


    def fit(self, data: pd.DataFrame, feature_cols, target_col: str):
       
        proc               = self._preprocess(data, feature_cols)
        self.feature_cols_ = list(feature_cols)   
        self.target_col_   = target_col

        self.root = self._build(proc, self.feature_cols_, target_col, depth=0)
        print("\n[A5] Custom Decision Tree trained!")

    def _predict_one(self, row: pd.Series, node: DecisionTreeNode):
        
        if node.label is not None:
            return node.label

        val   = row.get(node.feature)
        child = node.children.get(val)

        if child is None:
            return list(node.children.values())[0].label

        return self._predict_one(row, child)   

    def predict(self, data: pd.DataFrame) -> pd.Series:
        
        proc = self._preprocess(data, self.feature_cols_)
        return proc.apply(lambda r: self._predict_one(r, self.root), axis=1)

    def print_tree(self, node: DecisionTreeNode = None, indent: str = ""):
        
        node = node or self.root

        if node.label is not None:
            # Leaf node 
            print(f"{indent}-> LEAF  class = {node.label}")
            return

        # Internal node 
        print(f"{indent}[split on: {node.feature}]")
        for val, child in node.children.items():
            print(f"{indent}  |- bin = {val}")
            self.print_tree(child, indent + "  |   ")   

    def accuracy(self, data: pd.DataFrame, target_col: str) -> float:
        
        preds = self.predict(data)
        return (preds.values == data[target_col].values).mean()


#train
my_tree = MyDecisionTree(max_depth=5, binning_type="width", n_bins=4)
my_tree.fit(df, FEATURE_NAMES, TARGET_COL)

print("\n--- Custom Tree Structure (max_depth = 5) ---")
my_tree.print_tree()

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
acc = my_tree.accuracy(test_df, TARGET_COL)
print(f"\n   Custom Tree Test Accuracy : {acc * 100:.2f}%")


print("\n[A6] Training sklearn DT on full dataset for visualization...")

X_pca10_full = pca_feat.transform(X_full)          

X_tr, X_te, y_tr, y_te = train_test_split(
    X_pca10_full, y_full, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_tr, y_tr)
print(f"   Sklearn DT Test Accuracy : {clf.score(X_te, y_te) * 100:.2f}%")

fig, ax = plt.subplots(figsize=(24, 12))
plot_tree(
    clf,
    feature_names=FEATURE_NAMES,           
    class_names=["Class 0", "Class 1"],    
    filled=True,                            
    rounded=True,                           
    fontsize=9,
    ax=ax
)
plt.title("A6 - Decision Tree Visualization\n"
          "(Top-10 PCA Components of 384-dim Embeddings)", fontsize=14)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/A6_decision_tree.png", dpi=150)
plt.close()
print("   Saved -> A6_decision_tree.png")


print("\n[A7] Plotting Decision Boundary on top-2 PCA components...")

pca_2d = PCA(n_components=2, random_state=42)
X_2d   = pca_2d.fit_transform(X_full)   
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_2d, y_full, test_size=0.2, random_state=42)

clf_2d = DecisionTreeClassifier(max_depth=5, random_state=42)
clf_2d.fit(X_tr2, y_tr2)

fig, ax = plt.subplots(figsize=(10, 7))

DecisionBoundaryDisplay.from_estimator(
    clf_2d, X_tr2,
    response_method="predict",   
    cmap="coolwarm",             
    alpha=0.4,                   
    ax=ax
)

scatter = ax.scatter(
    X_2d[:, 0], X_2d[:, 1],
    c=y_full,                   
    cmap="coolwarm",
    edgecolors="k",
    s=8,                        
    alpha=0.6,
    linewidths=0.3
)
plt.colorbar(scatter, ax=ax, ticks=[0, 1], label="True Class")

ax.set_xlabel(
    f"PCA Component 1  ({pca_2d.explained_variance_ratio_[0] * 100:.1f}% variance)")
ax.set_ylabel(
    f"PCA Component 2  ({pca_2d.explained_variance_ratio_[1] * 100:.1f}% variance)")
ax.set_title(
    "A7 - Decision Boundary in 2-D PCA Space\n"
    "(384-dim embeddings projected to 2 principal components)")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/A7_decision_boundary.png", dpi=150)
plt.close()
print("   Saved -> A7_decision_boundary.png")

print("\nAll tasks A1-A7 completed successfully!")
