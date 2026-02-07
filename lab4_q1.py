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
X = np.load("X_features.npy")
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
print("TRAIN CONFUSION MATRIX:\n", cm_train)
print("Train Precision:", p_train)
print("Train Recall:", r_train)
print("Train F1-score:", f1_train)

print("\nTEST CONFUSION MATRIX:\n", cm_test)
print("Test Precision:", p_test)
print("Test Recall:", r_test)
print("Test F1-score:", f1_test)


