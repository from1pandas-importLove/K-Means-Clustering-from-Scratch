from pyexpat import features

import numpy as np
from sklearn.datasets import load_wine
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# scroll down to the bottom to implement your solution


class CustomKMeans:
    def __init__(self, k, max_iter=300, tol=1e-6):
        self.k = k
        self.centers = None
        self.max_iter = max_iter
        self.tol = tol


    def fit(self, X):
        n_samples, n_features = X.shape

        # rng = np.random.default_rng(seed=42)
        # initial_indices = rng.choice(n_samples, size=self.k, replace=False)
        # self.centers = X[initial_indices]

        self.centers = X[:self.k]

        for _ in range(self.max_iter):
            labels = self.predict(X)
            old_centers = self.centers.copy()
            self.centers = self.calculate_new_centers(X, labels)
            center_shift = np.linalg.norm(old_centers - self.centers)
            if center_shift < self.tol:
                break

        return self

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
        return np.argmin(distances, axis=1)

    def calculate_new_centers(self, X, labels):
        n_features = X.shape[1]
        centroids = np.zeros((self.k, n_features))

        for i in range(self.k):
            cluster_points = X[labels == i]
            centroids[i] = cluster_points.mean(axis=0)

        return centroids

    def inertia(self, X):
        labels = self.predict(X)
        distances = np.linalg.norm(X - self.centers[labels], axis=1)
        return np.sum(distances ** 2)

def plot_comparison(data: np.ndarray, predicted_clusters: np.ndarray, true_clusters: np.ndarray = None,
                    centers: np.ndarray = None, show: bool = True):

    # Use this function to visualize the results on Stage 6.

    if true_clusters is not None:
        plt.figure(figsize=(20, 10))

        plt.subplot(1, 2, 1)
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.title('Predicted clusters')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.grid()

        plt.subplot(1, 2, 2)
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=true_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.title('Ground truth')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.grid()
    else:
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=predicted_clusters, palette='deep')
        if centers is not None:
            sns.scatterplot(x=centers[:, 0], y=centers[:, 1], marker='X', color='k', s=200)
        plt.title('Predicted clusters')
        plt.xlabel('alcohol')
        plt.ylabel('malic_acid')
        plt.grid()

    plt.savefig('Visualization.png', bbox_inches='tight')
    if show:
        plt.show()


def calculate_inertias(model_class, X, k_range):
    inertia_list = []
    for k in k_range:
        model = model_class(k)
        model.fit(X)
        inertia_list.append(float(model.inertia(X)))
    return inertia_list

def

def plot_elbow_curve(k_values, inertia_list):
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertia_list, marker='o')
    plt.title('Elbow Method for Optimal k')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.xticks(k_values)
    plt.grid(True)
    plt.show()

if __name__ == '__main__':

    # Load data
    data = load_wine(as_frame=True, return_X_y=True)
    X_full, y_full = data

    # Permutate it to make things more interesting
    rnd = np.random.RandomState(42)
    permutations = rnd.permutation(len(X_full))
    X_full = X_full.iloc[permutations]
    y_full = y_full.iloc[permutations]

    # From dataframe to ndarray
    X_full = X_full.values
    y_full = y_full.values

    # Scale data
    scaler = StandardScaler()
    X_full = scaler.fit_transform(X_full)

    k_values = list(range(2, 11))
    inertias = calculate_inertias(CustomKMeans, X_full, k_values)
    print(inertias)
    # plot_elbow_curve(k_values, inertias)



