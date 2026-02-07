## ASSIGNMENT -4
#A1
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

#Function defenations

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_knn(X_train, y_train, k=3):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    return cm, precision, recall, f1

#main
# Load your dataset
X = np.load("X_telugu_embeddings.npy")
y = np.load("y_labels.npy")

# Split
X_train, X_test, y_train, y_test = split_data(X, y)

# Train
model = train_knn(X_train, y_train, k=3)

# Evaluate on TRAIN
cm_train, p_train, r_train, f1_train = evaluate_model(model, X_train, y_train)

# Evaluate on TEST
cm_test, p_test, r_test, f1_test = evaluate_model(model, X_test, y_test)

# Print results
print("Train confusion matrix:\n", cm_train)
print("Train Precision:", p_train)
print("Train Recall:", r_train)
print("Train F1-score:", f1_train)

print("\nTest confusion matrix:\n", cm_test)
print("Test Precision:", p_test)
print("Test Recall:", r_test)
print("Test F1-score:", f1_test)

##A2

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

#function
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_knn_regressor(X_train, y_train, k=3):
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    return model

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, r2


#main
# Load YOUR dataset
X = np.load("X_telugu_embeddings.npy")
y = np.load("y_labels.npy")

# Split
X_train, X_test, y_train, y_test = split_data(X, y)

# Train regression model
model = train_knn_regressor(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
mse, rmse, r2 = regression_metrics(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

##A3
import numpy as np
import matplotlib.pyplot as plt

#function
def generate_training_data(n=20):
    X = np.random.uniform(1, 10, (n, 2))  # 20 points, 2 features
    y = np.where(X[:, 0] + X[:, 1] > 10, 1, 0)  # class rule
    return X, y

#main
X_train, y_train = generate_training_data()

plt.figure(figsize=(7,6))
plt.scatter(X_train[y_train==0][:,0], X_train[y_train==0][:,1], color='blue', label='Class 0')
plt.scatter(X_train[y_train==1][:,0], X_train[y_train==1][:,1], color='red', label='Class 1')

plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.title("Training Data Scatter Plot")
plt.legend()
plt.grid(True)
plt.show()

##A4
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#function
def generate_training_data(n=20):
    X = np.random.uniform(1, 10, (n, 2))
    y = np.where(X[:, 0] + X[:, 1] > 10, 1, 0)  # simple rule for classes
    return X, y

def train_knn(X, y, k=3):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X, y)
    return model

def generate_test_grid():
    x = np.arange(0, 10, 0.1)
    y = np.arange(0, 10, 0.1)
    xx, yy = np.meshgrid(x, y)
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid

def plot_results(X_train, y_train, xx, yy, Z):
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.3)

    # training points
    plt.scatter(X_train[y_train==0][:,0], X_train[y_train==0][:,1], color='blue', label='Class 0')
    plt.scatter(X_train[y_train==1][:,0], X_train[y_train==1][:,1], color='red', label='Class 1')

    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.title("kNN Decision Boundary (k=3)")
    plt.legend()
    plt.show()

#main
X_train, y_train = generate_training_data()
model = train_knn(X_train, y_train)

xx, yy, grid = generate_test_grid()
Z = model.predict(grid)

plot_results(X_train, y_train, xx, yy, Z)

##A5
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#function
def generate_training_data(n=20):
    X = np.random.uniform(1, 10, (n, 2))
    y = np.where(X[:, 0] + X[:, 1] > 10, 1, 0)
    return X, y

def generate_test_grid():
    x = np.arange(0, 10, 0.1)
    y = np.arange(0, 10, 0.1)
    xx, yy = np.meshgrid(x, y)
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid

def plot_knn_boundary(X_train, y_train, k, xx, yy, grid):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    Z = model.predict(grid)

    plt.figure(figsize=(7,6))
    plt.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.3)
    plt.scatter(X_train[y_train==0][:,0], X_train[y_train==0][:,1], color='blue', label='Class 0')
    plt.scatter(X_train[y_train==1][:,0], X_train[y_train==1][:,1], color='red', label='Class 1')

    plt.title(f"kNN Decision Boundary (k={k})")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.legend()
    plt.show()

#main
X_train, y_train = generate_training_data()
xx, yy, grid = generate_test_grid()

for k in [1, 3, 5, 9]:
    plot_knn_boundary(X_train, y_train, k, xx, yy, grid)

##A6

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

#function
def load_and_filter_data(feature_file, label_file):
    X = np.load(feature_file)
    y = np.load(label_file)

    # Select only 2 features
    X = X[:, [0, 1]]

    # Select only first two classes
    classes = np.unique(y)
    class0, class1 = classes[0], classes[1]
    mask = (y == class0) | (y == class1)

    return X[mask], y[mask], class0, class1


def generate_grid(X):
    x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
    y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid


def plot_training_data(X, y, class0, class1):
    plt.figure(figsize=(7,6))
    plt.scatter(X[y==class0][:,0], X[y==class0][:,1], color='blue', label=f'Class {class0}')
    plt.scatter(X[y==class1][:,0], X[y==class1][:,1], color='red', label=f'Class {class1}')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Project Data Training Scatter Plot")
    plt.legend()
    plt.show()


def plot_decision_boundary(X, y, k, xx, yy, grid, class0, class1):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X, y)
    Z = model.predict(grid)

    plt.figure(figsize=(7,6))
    plt.contourf(xx, yy, Z.reshape(xx.shape), alpha=0.3)
    plt.scatter(X[y==class0][:,0], X[y==class0][:,1], color='blue', label=f'Class {class0}')
    plt.scatter(X[y==class1][:,0], X[y==class1][:,1], color='red', label=f'Class {class1}')
    plt.title(f"kNN Decision Boundary (k={k})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

#main
X, y, class0, class1 = load_and_filter_data("X_telugu_embeddings.npy", "y_labels.npy")

# A3-style plot
plot_training_data(X, y, class0, class1)

# Grid for A4/A5-style boundary
xx, yy, grid = generate_grid(X)

# A4 (k=3)
plot_decision_boundary(X, y, k=3, xx=xx, yy=yy, grid=grid, class0=class0, class1=class1)

# A5 (multiple k)
for k in [1, 5, 11, 21]:
    plot_decision_boundary(X, y, k, xx, yy, grid, class0, class1)


##A7
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

#function
def load_and_filter_data(feature_file, label_file):
    X = np.load(feature_file)
    y = np.load(label_file)

    # Use only 2 features
    X = X[:, [0, 1]]

    # Use only first two classes
    classes = np.unique(y)
    class0, class1 = classes[0], classes[1]
    mask = (y == class0) | (y == class1)

    return X[mask], y[mask]

def perform_grid_search(X_train, y_train):
    param_grid = {'n_neighbors': list(range(1, 31))}  # Try k from 1 to 30
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search

#main
X, y = load_and_filter_data("X_telugu_embeddings.npy", "y_labels.npy")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

grid_search = perform_grid_search(X_train, y_train)

print("Best k value:", grid_search.best_params_['n_neighbors'])
print("Best cross-validation score:", grid_search.best_score_)
