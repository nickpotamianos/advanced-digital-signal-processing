import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import CCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_curve, auc, precision_recall_curve)
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.model_selection import learning_curve
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from tqdm import tqdm

# Set plot style for academic presentation
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# 1) Debug: confirm working directory and files
print("Working directory:", os.getcwd())
print("Directory contents:", os.listdir())

# 2) Load splits.json
with open("splits.json", "r") as f:
    splits = json.load(f)


# 3) Load & center data
def load_set(key):
    X, y = [], []
    for e in splits[key]:
        img = np.array(Image.open(e["path"]), dtype=float).flatten()
        X.append(img)
        y.append(e["label"])
    return np.vstack(X), np.array(y)


X_train, y_train = load_set("train")
X_test, y_test = load_set("test")

# Get original image dimensions
first_img = Image.open(splits["train"][0]["path"])
width, height = first_img.size

# Compute mean face and center data
mean_face = X_train.mean(axis=0)
X_train_centered = X_train - mean_face
X_test_centered = X_test - mean_face

# 4) Encode labels for CCA and visualization
le = LabelEncoder().fit(y_train)
y_train_enc = le.transform(y_train).reshape(-1, 1)
y_test_enc = le.transform(y_test).reshape(-1, 1)
y_train_num = le.transform(y_train)
y_test_num = le.transform(y_test)

# Create results directory if it doesn't exist
results_dir = "visualization_results"
os.makedirs(results_dir, exist_ok=True)

# 5) PCA Analysis and Visualization
print("\n=== PCA Analysis ===")
n_components_full = min(X_train.shape[0], X_train.shape[1])
pca_full = PCA(n_components=n_components_full)
pca_full.fit(X_train_centered)

# 5.1) Eigenvalue Spectrum Visualization
plt.figure(figsize=(10, 6))
explained_var = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)
plt.plot(range(1, len(explained_var) + 1), cumulative_var, 'o-', markersize=8)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Explained Variance')
plt.axhline(y=0.99, color='g', linestyle='--', label='99% Explained Variance')
components_95 = np.where(cumulative_var >= 0.95)[0][0] + 1
components_99 = np.where(cumulative_var >= 0.99)[0][0] + 1
plt.axvline(x=components_95, color='r', linestyle='--')
plt.axvline(x=components_99, color='g', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('PCA Eigenvalue Spectrum Analysis')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{results_dir}/pca_eigenvalue_spectrum.png", dpi=300)
plt.close()

print(f"Components needed for 95% variance: {components_95}")
print(f"Components needed for 99% variance: {components_99}")

# 5.2) Top Eigenfaces visualization
n_eigenfaces = 20
pca = PCA(n_components=n_eigenfaces, svd_solver="randomized", whiten=True)
Z_train = pca.fit_transform(X_train_centered)
Z_test = pca.transform(X_test_centered)

# Eigenfaces grid
eigs = pca.components_.reshape((n_eigenfaces, height, width))
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
for i, ax in enumerate(axes.flat):
    if i < n_eigenfaces:
        ax.imshow(eigs[i], cmap="viridis")
        ax.set_title(f"Eigenface {i + 1}\nVar: {pca.explained_variance_ratio_[i]:.3f}")
        ax.axis("off")
    else:
        ax.axis("off")
plt.suptitle("PCA: Top 20 Eigenfaces with Explained Variance", fontsize=16)
plt.tight_layout()
plt.savefig(f"{results_dir}/eigenfaces_grid.png", dpi=300)
plt.close()

# 5.3) Face reconstruction with increasing eigenfaces
n_reconstructions = 5  # Number of faces to reconstruct
reconstruction_steps = [1, 5, 10, 20, 50, 100]  # Number of eigenfaces to use

# Get some test faces
test_faces_indices = np.random.choice(len(X_test), n_reconstructions, replace=False)
test_faces = X_test_centered[test_faces_indices]
test_labels = y_test[test_faces_indices]

# Create a larger PCA model for reconstruction
if max(reconstruction_steps) > n_eigenfaces:
    pca_recon = PCA(n_components=max(reconstruction_steps))
    pca_recon.fit(X_train_centered)
else:
    pca_recon = pca

# Plot reconstructions
fig, axes = plt.subplots(n_reconstructions, len(reconstruction_steps) + 1,
                         figsize=(16, 3 * n_reconstructions))

for i, face_idx in enumerate(range(n_reconstructions)):
    # Original face
    orig_face = test_faces[face_idx].reshape(height, width)
    axes[i, 0].imshow(orig_face, cmap='gray')
    axes[i, 0].set_title(f"Original ({test_labels[face_idx]})")
    axes[i, 0].axis('off')

    # Reconstructions with increasing eigenvectors
    for j, n_components in enumerate(reconstruction_steps):
        # Project to eigenspace and back
        reduced = pca_recon.transform(test_faces[face_idx].reshape(1, -1))[:, :n_components]
        reconstructed = np.dot(reduced, pca_recon.components_[:n_components, :]) + mean_face
        reconstructed = reconstructed.reshape(height, width)

        # Display reconstruction
        axes[i, j + 1].imshow(reconstructed, cmap='gray')
        axes[i, j + 1].set_title(f"{n_components} components")
        axes[i, j + 1].axis('off')

plt.suptitle("Face Reconstruction with Increasing Number of Eigenfaces", fontsize=16)
plt.tight_layout()
plt.savefig(f"{results_dir}/face_reconstruction.png", dpi=300)
plt.close()

# 5.4) 2D and 3D visualization of PCA-projected face data
# 2D projection (first 2 principal components)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(Z_train[:, 0], Z_train[:, 1], c=y_train_num,
                      cmap=ListedColormap(['#FF9999', '#66B2FF']),
                      s=100, alpha=0.7, edgecolors='w')
plt.scatter(Z_test[:, 0], Z_test[:, 1], c=y_test_num,
            cmap=ListedColormap(['#FF9999', '#66B2FF']),
            marker='x', s=150, label='Test samples')

# Add centroids
for i, gender in enumerate(le.classes_):
    centroid = np.mean(Z_train[y_train == gender, :2], axis=0)
    plt.scatter(centroid[0], centroid[1], s=300, c='black',
                marker='*', label=f'{gender} centroid', edgecolors='w')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA Projection of Face Data')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{results_dir}/pca_2d_projection.png", dpi=300)
plt.close()

# 3D projection (first 3 principal components)
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(Z_train[:, 0], Z_train[:, 1], Z_train[:, 2],
                     c=y_train_num, cmap=ListedColormap(['#FF9999', '#66B2FF']),
                     s=100, alpha=0.7)
ax.scatter(Z_test[:, 0], Z_test[:, 1], Z_test[:, 2],
           c=y_test_num, cmap=ListedColormap(['#FF9999', '#66B2FF']),
           marker='x', s=150)

# Add centroids in 3D
for i, gender in enumerate(le.classes_):
    centroid = np.mean(Z_train[y_train == gender, :3], axis=0)
    ax.scatter(centroid[0], centroid[1], centroid[2], s=300, c='black',
               marker='*', label=f'{gender} centroid')

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D PCA Projection of Face Data')
plt.legend()
plt.tight_layout()
plt.savefig(f"{results_dir}/pca_3d_projection.png", dpi=300)
plt.close()

# 5.5) Train a 1-NN classifier on the PCA projection
knn_pca = KNeighborsClassifier(n_neighbors=1).fit(Z_train, y_train)
pred_pca = knn_pca.predict(Z_test)
pca_accuracy = accuracy_score(y_test, pred_pca)
print(f"PCA + 1-NN Accuracy: {pca_accuracy:.4f}")

# 6) LDA Analysis and Visualization
print("\n=== LDA Analysis ===")
lda = LinearDiscriminantAnalysis(n_components=1)
L_train = lda.fit_transform(X_train_centered, y_train)
L_test = lda.transform(X_test_centered)

# 6.1) Visualize the Fisherface
fisherface = lda.scalings_[:, 0].reshape((height, width))
plt.figure(figsize=(8, 8))
plt.imshow(fisherface, cmap="plasma")
plt.title("LDA Fisherface for Gender Classification", fontsize=16)
plt.axis("off")
plt.colorbar(label='Weight')
plt.tight_layout()
plt.savefig(f"{results_dir}/fisherface.png", dpi=300)
plt.close()

# 6.2) Visualize LDA projection
plt.figure(figsize=(12, 6))
bins = np.linspace(min(np.min(L_train), np.min(L_test)),
                   max(np.max(L_train), np.max(L_test)), 50)

plt.hist(L_train[y_train == 'female'], bins=bins, alpha=0.5, color='red',
         label='Female (Train)')
plt.hist(L_train[y_train == 'male'], bins=bins, alpha=0.5, color='blue',
         label='Male (Train)')

plt.hist(L_test[y_test == 'female'], bins=bins, alpha=0.8, color='darkred',
         label='Female (Test)', histtype='step', linewidth=2)
plt.hist(L_test[y_test == 'male'], bins=bins, alpha=0.8, color='darkblue',
         label='Male (Test)', histtype='step', linewidth=2)

# Decision boundary
threshold = (np.mean(L_train[y_train == 'female']) + np.mean(L_train[y_train == 'male'])) / 2
plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2,
            label='Decision Boundary')

plt.xlabel('LDA Projection Value')
plt.ylabel('Frequency')
plt.title('LDA Projection Distribution for Gender Classification', fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{results_dir}/lda_projection.png", dpi=300)
plt.close()

# 6.3) Train a 1-NN classifier on the LDA projection
knn_lda = KNeighborsClassifier(n_neighbors=1).fit(L_train, y_train)
pred_lda = knn_lda.predict(L_test)
lda_accuracy = accuracy_score(y_test, pred_lda)
print(f"LDA + 1-NN Accuracy: {lda_accuracy:.4f}")

# 7) Centroid Classifier
print("\n=== Centroid Classifier Analysis ===")
male_centroid = X_train_centered[y_train == "male"].mean(axis=0)
female_centroid = X_train_centered[y_train == "female"].mean(axis=0)

# 7.1) Visualize class mean faces (centroids)
male_mean_face = male_centroid.reshape((height, width))
female_mean_face = female_centroid.reshape((height, width))
diff_face = (female_mean_face - male_mean_face)
normalized_diff = (diff_face - diff_face.min()) / (diff_face.max() - diff_face.min())

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(male_mean_face, cmap="gray")
axes[0].set_title("Mean Male Face", fontsize=14)
axes[0].axis("off")

axes[1].imshow(female_mean_face, cmap="gray")
axes[1].set_title("Mean Female Face", fontsize=14)
axes[1].axis("off")

axes[2].imshow(normalized_diff, cmap="RdBu_r")
axes[2].set_title("Gender Difference Map", fontsize=14)
axes[2].axis("off")

plt.suptitle("Class Centroids and Difference Analysis", fontsize=16)
plt.tight_layout()
plt.savefig(f"{results_dir}/centroid_analysis.png", dpi=300)
plt.close()


# 7.2) Predict with centroid classifier
def predict_centroid(X):
    d_m = np.linalg.norm(X - male_centroid, axis=1)
    d_f = np.linalg.norm(X - female_centroid, axis=1)
    return np.where(d_m < d_f, "male", "female")


pred_centroid = predict_centroid(X_test_centered)
centroid_accuracy = accuracy_score(y_test, pred_centroid)
print(f"Centroid Classifier Accuracy: {centroid_accuracy:.4f}")

# 8) K-means Clustering Analysis (as specifically requested)
print("\n=== K-means Clustering Analysis ===")

# 8.1) Apply k-means to PCA-projected data
n_clusters = 2  # For gender classification
kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_pca.fit(Z_train)


# 8.2) Visualize k-means convergence
def run_kmeans_with_history(X, n_clusters, max_iters=10):
    # Initialize centroids
    np.random.seed(42)
    indices = np.random.choice(X.shape[0], n_clusters, replace=False)
    centroids = X[indices]

    history = [centroids.copy()]
    labels_history = []
    inertia_history = []

    for iteration in range(max_iters):
        # Assign labels based on closest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        labels_history.append(labels.copy())

        # Calculate inertia
        inertia = 0
        for i in range(n_clusters):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - centroids[i]) ** 2)
        inertia_history.append(inertia)

        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) if np.sum(labels == i) > 0
                                  else centroids[i] for i in range(n_clusters)])

        # Check for convergence
        if np.allclose(new_centroids, centroids):
            history.append(new_centroids.copy())
            break

        centroids = new_centroids
        history.append(centroids.copy())

    return history, labels_history, inertia_history


# Use the first 2 PCA components for visualization
X_pca_2d = Z_train[:, :2]
centroid_history, labels_history, inertia_history = run_kmeans_with_history(X_pca_2d, n_clusters)

# Plot k-means convergence
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
iterations_to_plot = [0, 1, len(centroid_history) - 1]  # Initial, step 1, final

for i, ax_col in enumerate(iterations_to_plot):
    if ax_col < len(labels_history):
        # Top row: scatter plot with centroids
        axes[0, i].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=labels_history[ax_col],
                           cmap='viridis', s=50, alpha=0.7)
        axes[0, i].scatter(centroid_history[ax_col][:, 0], centroid_history[ax_col][:, 1],
                           c='red', marker='X', s=200, edgecolors='black')
        axes[0, i].set_title(f'Iteration {ax_col}', fontsize=14)
        axes[0, i].set_xlabel('Principal Component 1')
        axes[0, i].set_ylabel('Principal Component 2')
        axes[0, i].grid(True)

# Bottom left: centroid movement
for j, cluster in enumerate(range(n_clusters)):
    cluster_centroids = np.array([ch[cluster] for ch in centroid_history])
    axes[1, 0].plot(cluster_centroids[:, 0], cluster_centroids[:, 1], 'o-',
                    markersize=10, label=f'Cluster {cluster}')
    # Mark start and end
    axes[1, 0].plot(cluster_centroids[0, 0], cluster_centroids[0, 1], 'o',
                    markersize=15, color='green', label='_nolegend_')
    axes[1, 0].plot(cluster_centroids[-1, 0], cluster_centroids[-1, 1], 'o',
                    markersize=15, color='red', label='_nolegend_')

axes[1, 0].set_title('Centroid Movement During Iterations', fontsize=14)
axes[1, 0].set_xlabel('Principal Component 1')
axes[1, 0].set_ylabel('Principal Component 2')
axes[1, 0].grid(True)
axes[1, 0].legend()

# Bottom middle: Inertia (sum of squared distances)
axes[1, 1].plot(range(len(inertia_history)), inertia_history, 'o-', markersize=10)
axes[1, 1].set_title('K-means Convergence: Inertia', fontsize=14)
axes[1, 1].set_xlabel('Iteration')
axes[1, 1].set_ylabel('Inertia (Sum of Squared Distances)')
axes[1, 1].grid(True)

# Bottom right: Compare k-means clusters with true labels
axes[1, 2].scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y_train_num,
                   cmap=ListedColormap(['#FF9999', '#66B2FF']), s=50, alpha=0.7)
axes[1, 2].set_title('True Gender Labels', fontsize=14)
axes[1, 2].set_xlabel('Principal Component 1')
axes[1, 2].set_ylabel('Principal Component 2')
axes[1, 2].grid(True)

plt.suptitle('K-means Clustering Analysis on PCA-projected Face Data', fontsize=16)
plt.tight_layout()
plt.savefig(f"{results_dir}/kmeans_convergence.png", dpi=300)
plt.close()

# 8.3) Evaluate k-means clusters against true labels
pca_clusters = kmeans_pca.predict(Z_train)
contingency_table = confusion_matrix(y_train_num, pca_clusters)

# We need to align cluster labels with true labels
if contingency_table[0, 0] + contingency_table[1, 1] < contingency_table[0, 1] + contingency_table[1, 0]:
    # Swap cluster labels if they're inverted
    pca_clusters = 1 - pca_clusters

print("K-means Clustering Evaluation:")
print(confusion_matrix(y_train_num, pca_clusters))
print(f"Clustering accuracy: {accuracy_score(y_train_num, pca_clusters):.4f}")

# 8.4) Apply k-means to LDA projection
kmeans_lda = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_lda.fit(L_train)
lda_clusters = kmeans_lda.predict(L_train)

# Align cluster labels with true labels
contingency_table = confusion_matrix(y_train_num, lda_clusters)
if contingency_table[0, 0] + contingency_table[1, 1] < contingency_table[0, 1] + contingency_table[1, 0]:
    lda_clusters = 1 - lda_clusters

print("\nLDA K-means Clustering Evaluation:")
print(confusion_matrix(y_train_num, lda_clusters))
print(f"Clustering accuracy: {accuracy_score(y_train_num, lda_clusters):.4f}")

# 8.5) K-means with different K values (Elbow Method)
max_k = 10
inertias = []
silhouette_scores = []
from sklearn.metrics import silhouette_score

for k in range(2, max_k + 1):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(Z_train)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(Z_train, kmeans.labels_))

# Plot elbow method
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Inertia plot
ax1.plot(range(2, max_k + 1), inertias, 'bo-', markersize=8)
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method for Optimal k')
ax1.grid(True)

# Silhouette score plot
ax2.plot(range(2, max_k + 1), silhouette_scores, 'ro-', markersize=8)
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Method for Optimal k')
ax2.grid(True)

plt.suptitle('Determining Optimal Number of Clusters', fontsize=16)
plt.tight_layout()
plt.savefig(f"{results_dir}/kmeans_elbow_method.png", dpi=300)
plt.close()

# 9) CCA (Canonical Correlation Analysis)
print("\n=== CCA Analysis ===")
cca = CCA(n_components=1)
U_train, _ = cca.fit_transform(X_train_centered, y_train_enc)
U_test, _ = cca.transform(X_test_centered, y_test_enc)

# 9.1) Visualize CCA projection
plt.figure(figsize=(12, 6))
bins = np.linspace(min(np.min(U_train), np.min(U_test)),
                   max(np.max(U_train), np.max(U_test)), 50)

plt.hist(U_train[y_train == 'female'], bins=bins, alpha=0.5, color='red',
         label='Female (Train)')
plt.hist(U_train[y_train == 'male'], bins=bins, alpha=0.5, color='blue',
         label='Male (Train)')

plt.hist(U_test[y_test == 'female'], bins=bins, alpha=0.8, color='darkred',
         label='Female (Test)', histtype='step', linewidth=2)
plt.hist(U_test[y_test == 'male'], bins=bins, alpha=0.8, color='darkblue',
         label='Male (Test)', histtype='step', linewidth=2)

plt.xlabel('CCA Projection Value')
plt.ylabel('Frequency')
plt.title('CCA Projection Distribution for Gender Classification', fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{results_dir}/cca_projection.png", dpi=300)
plt.close()

# 9.2) Visualize canonical variate
canon_var = cca.x_weights_[:, 0].reshape((height, width))
plt.figure(figsize=(8, 8))
plt.imshow(canon_var, cmap="coolwarm")
plt.title("CCA: Canonical Variate for Gender Classification", fontsize=16)
plt.colorbar(label='Weight')
plt.axis("off")
plt.tight_layout()
plt.savefig(f"{results_dir}/cca_variate.png", dpi=300)
plt.close()

# 9.3) Train a 1-NN classifier on the CCA projection
knn_cca = KNeighborsClassifier(n_neighbors=1).fit(U_train, y_train)
pred_cca = knn_cca.predict(U_test)
cca_accuracy = accuracy_score(y_test, pred_cca)
print(f"CCA + 1-NN Accuracy: {cca_accuracy:.4f}")

# For the LDA learning curve issue, replace that section with:

# 10) Learning Curves Visualization - With custom CV to ensure both classes in each fold
print("\n=== Learning Curves Analysis ===")

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5,
                        n_jobs=None, train_sizes=np.linspace(.3, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title(title)
    plt.grid(True)

    # Use stratified CV to ensure both classes in each fold
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv_strategy, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


# Learning curve for PCA+KNN
pca_knn_pipe = Pipeline([
    ('pca', PCA(n_components=20, whiten=True)),
    ('knn', KNeighborsClassifier(n_neighbors=1))
])
plot_learning_curve(pca_knn_pipe, "Learning Curve: PCA + 1-NN",
                    X_train_centered, y_train_num, ylim=(0.7, 1.01), cv=5)
plt.savefig(f"{results_dir}/pca_knn_learning_curve.png", dpi=300)
plt.close()

# Fixed Learning curve for LDA+KNN with stratified CV
lda_knn_pipe = Pipeline([
    ('lda', LinearDiscriminantAnalysis(n_components=1, solver='svd')),  # Use SVD solver
    ('knn', KNeighborsClassifier(n_neighbors=1))
])
plot_learning_curve(lda_knn_pipe, "Learning Curve: LDA + 1-NN",
                    X_train_centered, y_train_num, ylim=(0.7, 1.01), cv=5)
plt.savefig(f"{results_dir}/lda_knn_learning_curve.png", dpi=300)
plt.close()


# Add these additional academic visualizations:

# 1. Decision Boundary Visualization for PCA (2D projection)
def plot_decision_boundary(X, y, classifier, title):
    # Create mesh grid
    h = 0.02  # step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict class for each point in mesh
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array([1 if z == 'female' else 0 for z in Z]).reshape(xx.shape)

    # Plot decision boundary
    plt.figure(figsize=(12, 10))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['#AAAAFF', '#FFAAAA']))

    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y,
                          cmap=ListedColormap(['#FF9999', '#66B2FF']),
                          edgecolors='k', s=80)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(title)
    plt.grid(True)
    plt.colorbar(scatter, ticks=[0, 1], label='Gender')
    plt.tight_layout()
    return plt


# Plot decision boundary for PCA (first 2 components)
X_train_pca_2d = Z_train[:, :2]
knn_pca_2d = KNeighborsClassifier(n_neighbors=1).fit(X_train_pca_2d, y_train)
plot_decision_boundary(X_train_pca_2d, y_train_num, knn_pca_2d,
                       "Decision Boundary: PCA + 1-NN (2D Projection)")
plt.savefig(f"{results_dir}/pca_decision_boundary.png", dpi=300)
plt.close()

# 2. Feature Importance Analysis
print("\n=== Feature Importance Analysis ===")

# For PCA - visualize component importance
plt.figure(figsize=(10, 6))
importance = pca.explained_variance_ratio_[:20]
plt.bar(range(1, len(importance) + 1), importance, alpha=0.7)
plt.step(range(1, len(importance) + 1), np.cumsum(importance), where='mid',
         color='red', alpha=0.7)
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('PCA Component Importance')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{results_dir}/pca_component_importance.png", dpi=300)
plt.close()

# For LDA - visualize coefficient importance (fisherface)
lda_coef = np.abs(lda.scalings_[:, 0])
# Get the top 50 most important pixels
top_indices = np.argsort(lda_coef)[-50:]

# Create a heatmap of the importance on a face image
importance_map = np.zeros_like(male_mean_face)
for idx in top_indices:
    # Convert 1D index to 2D coordinates
    y, x = np.unravel_index(idx, (height, width))
    importance_map[y, x] = 1

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(male_mean_face, cmap='gray')
plt.title('Mean Face')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(male_mean_face, cmap='gray', alpha=0.7)
plt.imshow(importance_map, cmap='hot', alpha=0.5)
plt.title('Top 50 Most Important Pixels in LDA')
plt.axis('off')
plt.tight_layout()
plt.savefig(f"{results_dir}/lda_feature_importance.png", dpi=300)
plt.close()

# 3. ROC Curve Analysis for Binary Gender Classification
print("\n=== ROC Curve Analysis ===")


def roc_curve_analysis(classifiers_dict, X_test_dict, y_test_true):
    """Plot ROC curves for multiple classifiers"""
    plt.figure(figsize=(10, 8))

    for name, (classifier, X_test) in classifiers_dict.items():
        if hasattr(classifier, 'predict_proba'):
            # For classifiers that directly support probability estimation
            y_score = classifier.predict_proba(X_test)[:, 1]
        elif hasattr(classifier, 'decision_function'):
            # For classifiers with decision_function
            y_score = classifier.decision_function(X_test)
        else:
            # For LDA, we need to get the projection and use it as score
            if name == 'LDA + 1-NN':
                # Calculate distance to means
                y_score = X_test.ravel()
            elif name == 'Centroid':
                # Calculate distance ratio
                d_m = np.linalg.norm(X_test_centered - male_centroid, axis=1)
                d_f = np.linalg.norm(X_test_centered - female_centroid, axis=1)
                y_score = d_m / (d_m + d_f)  # Score between 0 and 1
            else:
                # We can use the distance to neighbors for KNN methods
                dist = np.array([np.min(np.linalg.norm(X_test[i] - X_train_dict[name][y_train_num == j],
                                                       axis=1))
                                 for i in range(len(X_test)) for j in [0, 1]]).reshape(len(X_test), 2)
                y_score = dist[:, 0] / (dist[:, 0] + dist[:, 1])

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test_true, y_score)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')

    # Plot random classifier
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/roc_curve_comparison.png", dpi=300)
    plt.close()


# Prepare classifiers and test data for ROC analysis
X_train_dict = {
    'PCA + 1-NN': Z_train,
    'LDA + 1-NN': L_train,
    'CCA + 1-NN': U_train
}

classifiers_dict = {
    'PCA + 1-NN': (knn_pca, Z_test),
    'LDA + 1-NN': (knn_lda, L_test),
    'CCA + 1-NN': (knn_cca, U_test)
}

# Perform ROC analysis
roc_curve_analysis(classifiers_dict, X_train_dict, y_test_num)

# 4. Learning Rate Convergence for Dimension Reduction Methods
print("\n=== Dimension Reduction Convergence Analysis ===")


# Function to compute reconstruction error for different numbers of PCA components
def compute_pca_reconstruction_error(X_train, X_test, max_components=100):
    pca_error = []
    components = np.arange(1, max_components + 1, 5)  # Evaluate at intervals

    for n_comp in components:
        pca_model = PCA(n_components=n_comp)
        pca_model.fit(X_train)

        # Transform and reconstruct
        X_test_transformed = pca_model.transform(X_test)
        X_test_reconstructed = pca_model.inverse_transform(X_test_transformed)

        # Compute MSE
        mse = np.mean((X_test - X_test_reconstructed) ** 2)
        pca_error.append(mse)

    return components, pca_error


# Calculate reconstruction error
max_comp = 100
components, pca_errors = compute_pca_reconstruction_error(X_train_centered, X_test_centered, max_comp)

# Plot reconstruction error
plt.figure(figsize=(10, 6))
plt.plot(components, pca_errors, 'o-', markersize=8, label='PCA Reconstruction Error')

# Add vertical lines for 95% and 99% variance explained
plt.axvline(x=components_95, color='r', linestyle='--', label=f'95% variance ({components_95} components)')
plt.axvline(x=components_99, color='g', linestyle='--', label=f'99% variance ({components_99} components)')

plt.xlabel('Number of Principal Components')
plt.ylabel('Mean Squared Reconstruction Error')
plt.title('PCA Reconstruction Error vs. Number of Components')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(f"{results_dir}/pca_reconstruction_error.png", dpi=300)
plt.close()

# 5. Clustering quality metrics
print("\n=== Clustering Quality Metrics ===")

from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

# Calculate metrics for different k values
k_range = range(2, 11)
ch_scores = []
db_scores = []
sil_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(Z_train)

    # Calculate quality metrics
    ch_scores.append(calinski_harabasz_score(Z_train, clusters))
    db_scores.append(davies_bouldin_score(Z_train, clusters))
    sil_scores.append(silhouette_score(Z_train, clusters))

# Plot clustering quality metrics
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# Calinski-Harabasz Index (higher is better)
ax1.plot(k_range, ch_scores, 'o-', markersize=8)
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Calinski-Harabasz Index')
ax1.set_title('Calinski-Harabasz Index (higher is better)')
ax1.grid(True)

# Davies-Bouldin Index (lower is better)
ax2.plot(k_range, db_scores, 'o-', markersize=8)
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Davies-Bouldin Index')
ax2.set_title('Davies-Bouldin Index (lower is better)')
ax2.grid(True)

# Silhouette Score (higher is better)
ax3.plot(k_range, sil_scores, 'o-', markersize=8)
ax3.set_xlabel('Number of Clusters (k)')
ax3.set_ylabel('Silhouette Score')
ax3.set_title('Silhouette Score (higher is better)')
ax3.grid(True)

plt.suptitle('Clustering Quality Metrics for Different k Values', fontsize=16)
plt.tight_layout()
plt.savefig(f"{results_dir}/clustering_quality_metrics.png", dpi=300)
plt.close()

# 11) t-SNE Visualization for high-dimensional data
print("\n=== t-SNE Visualization of Face Space ===")
# Use PCA first to reduce dimensionality for faster t-SNE
pca_50 = PCA(n_components=50)
X_train_pca_50 = pca_50.fit_transform(X_train_centered)

# Run t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
X_train_tsne = tsne.fit_transform(X_train_pca_50)

# Visualize t-SNE results
plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1],
                      c=y_train_num, cmap=ListedColormap(['#FF9999', '#66B2FF']),
                      s=100, alpha=0.7)
plt.title('t-SNE Visualization of Face Data', fontsize=16)
plt.colorbar(scatter, ticks=[0, 1], label='Gender')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{results_dir}/tsne_visualization.png", dpi=300)
plt.close()

# 12) Visualize correctly and incorrectly classified samples
print("\n=== Misclassification Analysis ===")


def plot_classification_results(X_test, y_test, y_pred, method_name):
    # Find correct and incorrect predictions
    correct = y_pred == y_test
    incorrect = ~correct

    # Get some examples
    n_examples = min(5, sum(incorrect))
    incorrect_indices = np.where(incorrect)[0][:n_examples]

    if len(incorrect_indices) > 0:
        fig, axes = plt.subplots(n_examples, 2, figsize=(10, 3 * n_examples))
        if n_examples == 1:
            axes = axes.reshape(1, 2)

        for i, idx in enumerate(incorrect_indices):
            # Original image
            img = X_test[idx].reshape(height, width) + mean_face.reshape(height, width)
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title(f"True: {y_test[idx]}, Predicted: {y_pred[idx]}")
            axes[i, 0].axis('off')

            # Difference from centroid
            true_centroid = female_centroid if y_test[idx] == 'female' else male_centroid
            pred_centroid = female_centroid if y_pred[idx] == 'female' else male_centroid

            diff = (X_test[idx] - pred_centroid).reshape(height, width)
            diff = (diff - diff.min()) / (diff.max() - diff.min())

            axes[i, 1].imshow(diff, cmap='coolwarm')
            axes[i, 1].set_title("Difference from predicted centroid")
            axes[i, 1].axis('off')

        plt.suptitle(f"Misclassified Examples - {method_name}", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/{method_name.lower().replace(' ', '_')}_misclassified.png", dpi=300)
        plt.close()
    else:
        print(f"No misclassifications for {method_name}!")


# Visualize misclassifications for each method
plot_classification_results(X_test, y_test, pred_pca, "PCA + 1-NN")
plot_classification_results(X_test, y_test, pred_lda, "LDA + 1-NN")
plot_classification_results(X_test, y_test, pred_centroid, "Centroid Classifier")
plot_classification_results(X_test, y_test, pred_cca, "CCA + 1-NN")

# 13) Comparative performance analysis
print("\n=== Comparative Performance Analysis ===")

# Define all methods
methods = {
    "PCA + 1-NN": pred_pca,
    "LDA + 1-NN": pred_lda,
    "Centroid": pred_centroid,
    "CCA + 1-NN": pred_cca
}

# Calculate accuracy for each method
accuracies = {name: accuracy_score(y_test, pred) for name, pred in methods.items()}

# Generate comparison bar plot
plt.figure(figsize=(12, 6))
bars = plt.bar(accuracies.keys(), accuracies.values(), color=['skyblue', 'lightgreen', 'salmon', 'orchid'])

# Add accuracy values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
             f'{height:.4f}', ha='center', va='bottom', fontsize=12)

plt.ylim(0, 1.1)
plt.ylabel('Accuracy')
plt.title('Comparison of Classification Methods for Gender Recognition', fontsize=16)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{results_dir}/method_comparison.png", dpi=300)
plt.close()

# Generate confusion matrices for all methods
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for i, (name, pred) in enumerate(methods.items()):
    cm = confusion_matrix(y_test, pred, labels=le.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(ax=axes[i], cmap='Blues', values_format='d')
    axes[i].set_title(f"{name} (Accuracy: {accuracies[name]:.4f})", fontsize=14)

plt.suptitle('Confusion Matrices for All Classification Methods', fontsize=16)
plt.tight_layout()
plt.savefig(f"{results_dir}/all_confusion_matrices.png", dpi=300)
plt.close()

# Print summary
print("\nAccuracy Summary:")
for name, acc in accuracies.items():
    print(f"{name}: {acc:.4f}")

print("\nAll visualizations saved to directory:", results_dir)