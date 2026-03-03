# =========================
# IMPORTS
# =========================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

# =========================
# METRIC FUNCTIONS
# =========================

def calculate_metrics(y_true, y_pred):
    """Returns MSE, RMSE, MAPE and R2 score"""
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    return mse, rmse, mape, r2


# =========================
# LINEAR REGRESSION FUNCTIONS
# =========================

def train_linear_regression(X_train, y_train):
    """Train linear regression model"""
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model on train and test data"""
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    return train_metrics, test_metrics


# =========================
# K-MEANS FUNCTIONS
# =========================

def perform_kmeans(X, k):
    """Perform k-means clustering"""
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X)
    
    return kmeans


def evaluate_clustering(X, labels):
    """Calculate Silhouette, CH and DB scores"""
    
    sil_score = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    db_score = davies_bouldin_score(X, labels)
    
    return sil_score, ch_score, db_score


def evaluate_k_range(X, k_values):
    """Evaluate clustering for multiple k values"""
    
    silhouette_scores = []
    ch_scores = []
    db_scores = []
    distortions = []
    
    for k in k_values:
        kmeans = perform_kmeans(X, k)
        labels = kmeans.labels_
        
        sil, ch, db = evaluate_clustering(X, labels)
        
        silhouette_scores.append(sil)
        ch_scores.append(ch)
        db_scores.append(db)
        distortions.append(kmeans.inertia_)
    
    return silhouette_scores, ch_scores, db_scores, distortions


# =========================
# MAIN PROGRAM
# =========================

if __name__ == "__main__":
    
    # Load dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target   # numerical target (classification converted to regression)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # =========================
    # A1 & A2 (Single Attribute)
    # =========================
    
    X_train_single = X_train[['mean radius']]
    X_test_single = X_test[['mean radius']]
    
    model_single = train_linear_regression(X_train_single, y_train)
    train_metrics_single, test_metrics_single = evaluate_model(
        model_single, X_train_single, y_train, X_test_single, y_test
    )
    
    print("Single Attribute - Train Metrics (MSE, RMSE, MAPE, R2):")
    print(train_metrics_single)
    
    print("Single Attribute - Test Metrics (MSE, RMSE, MAPE, R2):")
    print(test_metrics_single)
    
    
    # =========================
    # A3 (Multiple Attributes)
    # =========================
    
    model_multi = train_linear_regression(X_train, y_train)
    train_metrics_multi, test_metrics_multi = evaluate_model(
        model_multi, X_train, y_train, X_test, y_test
    )
    
    print("\nMultiple Attributes - Train Metrics:")
    print(train_metrics_multi)
    
    print("Multiple Attributes - Test Metrics:")
    print(test_metrics_multi)
    
    
    # =========================
    # A4 & A5 (K-Means k=2)
    # =========================
    
    X_cluster = X_train.copy()  # remove target
    
    kmeans_2 = perform_kmeans(X_cluster, 2)
    sil, ch, db = evaluate_clustering(X_cluster, kmeans_2.labels_)
    
    print("\nClustering Scores for k=2")
    print("Silhouette:", sil)
    print("CH Score:", ch)
    print("DB Index:", db)
    
    
    # =========================
    # A6 (Different k values)
    # =========================
    
    k_values = range(2, 10)
    sil_scores, ch_scores, db_scores, distortions = evaluate_k_range(
        X_cluster, k_values
    )
    
    # Plot scores vs k
    plt.figure()
    plt.plot(k_values, sil_scores)
    plt.title("Silhouette Score vs k")
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.show()
    
    plt.figure()
    plt.plot(k_values, ch_scores)
    plt.title("CH Score vs k")
    plt.xlabel("k")
    plt.ylabel("CH Score")
    plt.show()
    
    plt.figure()
    plt.plot(k_values, db_scores)
    plt.title("DB Index vs k")
    plt.xlabel("k")
    plt.ylabel("DB Index")
    plt.show()
    
    
    # =========================
    # A7 (Elbow Method)
    # =========================
    
    plt.figure()
    plt.plot(k_values, distortions)
    plt.title("Elbow Plot")
    plt.xlabel("k")
    plt.ylabel("Distortion (Inertia)")
    plt.show()