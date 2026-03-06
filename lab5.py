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

#metrics function
def metrics(y_true, y_pred):    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    
    return mse, rmse, mape, r2


#linear regression 
def train_lr(X_train, y_train):    
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def eval_model(model, X_train, y_train, X_test, y_test):    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_metrics = metrics(y_train, y_train_pred)
    test_metrics = metrics(y_test, y_test_pred)
    
    return train_metrics, test_metrics


#k-means
def perform_kmeans(X, k):    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
    kmeans.fit(X)
    
    return kmeans


def eval_clustering(X, labels):    
    sil_score = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    db_score = davies_bouldin_score(X, labels)
    
    return sil_score, ch_score, db_score


def eval_k_range(X, k_values):    
    silhouette_scores = []
    ch_scores = []
    db_scores = []
    distortions = []
    
    for k in k_values:
        kmeans = perform_kmeans(X, k)
        labels = kmeans.labels_
        
        sil, ch, db = eval_clustering(X, labels)
        
        silhouette_scores.append(sil)
        ch_scores.append(ch)
        db_scores.append(db)
        distortions.append(kmeans.inertia_)
    
    return silhouette_scores, ch_scores, db_scores, distortions


#main
if __name__ == "__main__":
    
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    #test and train split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    #single attributes
    X_train_single = X_train[['mean radius']]
    X_test_single = X_test[['mean radius']]
    
    model_s = train_lr(X_train_single, y_train)
    train_metrics_s, test_metrics_s = eval_model(
        model_s, X_train_single, y_train, X_test_single, y_test
    )
    
    print("train Single attributes:")
    print(train_metrics_s)
    
    print("test Single attributes:")
    print(test_metrics_s)
    #multiple attributes
    model_multi = train_lr(X_train, y_train)
    train_metrics_m, test_metrics_m = eval_model(
        model_multi, X_train, y_train, X_test, y_test
    )
    
    print("\train Multiple attributes:")
    print(train_metrics_m)
    
    print("test Multiple attributes:")
    print(test_metrics_m)
    #k-means
    X_cluster = X_train.copy()  
    
    kmeans_2 = perform_kmeans(X_cluster, 2)
    sil, ch, db = eval_clustering(X_cluster, kmeans_2.labels_)
    
    print("\nClustering Scores for k=2")
    print("Silhouette:", sil)
    print("CH Score:", ch)
    print("DB Index:", db)
        
    k_values = range(2, 10)
    sil_scores, ch_scores, db_scores, distortions = eval_k_range(
        X_cluster, k_values
    )
    
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
    #elbow
    plt.figure()
    plt.plot(k_values, distortions)
    plt.title("Elbow Plot")
    plt.xlabel("k")
    plt.ylabel("Distortion")
    plt.show()
