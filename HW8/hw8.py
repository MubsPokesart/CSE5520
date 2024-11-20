import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.stats.multitest import multipletests
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

START_DATE = "2009-10-27" # First day filtering
END_DATE = "2019-10-22" # Last day filtering

# Teams in NBA based on dataset
WESTERN_CONFERENCE_TEAMS = ['Portland Trail Blazers', 'Los Angeles Lakers', 'Dallas Mavericks', 'Golden State Warriors', 'Denver Nuggets',
                            'Los Angeles Clippers', 'San Antonio Spurs', 'Minnesota Timberwolves', 'Memphis Grizzlies', 'New Orleans Hornets',
                            'Phoenix Suns', 'Oklahoma City Thunder', 'Utah Jazz', 'Houston Rockets', 'Sacramento Kings', 'LA Clippers', 'New Orleans Pelicans']

EASTERN_CONFERENCE_TEAMS = ['Cleveland Cavaliers', 'Atlanta Hawks', 'Miami Heat', 'Boston Celtics',  'Orlando Magic', 'Toronto Raptors',
                            'Chicago Bulls', 'New Jersey Nets', 'Detroit Pistons', 'Charlotte Bobcats', 'Philadelphia 76ers', 'Indiana Pacers', 
                            'Washington Wizards', 'New York Knicks', 'Milwaukee Bucks', 'Brooklyn Nets', 'Charlotte Hornets']
ALL_NBA_TEAMS = WESTERN_CONFERENCE_TEAMS + EASTERN_CONFERENCE_TEAMS
#
METRICS = ['pts_home', 'fg_pct_home', 'fg3_pct_home', 'ft_pct_home', 'reb_home', 'ast_home', 'stl_home', 'blk_home', 'tov_home']

#   Create DataFrame for summary and filter data
summary_df = pd.read_csv("game.csv")
summary_df['game_date'] = pd.to_datetime(summary_df['game_date'])
range_dataframe = summary_df[(summary_df['game_date'] >= START_DATE) & (summary_df['game_date'] < END_DATE)]
range_dataframe = range_dataframe[range_dataframe['team_name_home'].isin(ALL_NBA_TEAMS)]

# STEP 1 -**Step 1.** Plot your 2D data (scatter plot). Determine the number of clusters k **(extra credit)** You can use silhouette method or elbow method to find the optimum k.
X = range_dataframe[['pts_home', 'fg_pct_home']].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. Create scatter plot of original data
plt.figure(figsize=(10, 6))
plt.scatter(X['pts_home'], X['fg_pct_home'], alpha=0.5)
plt.title('NBA Home Games: Points vs Field Goal Percentage')
plt.xlabel('Points Scored (Home)')
plt.ylabel('Field Goal Percentage (Home)')
plt.grid(True, alpha=0.3)
plt.show()

inertias = []
K = range(1, 11)  # Testing k from 1 to 10

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K, inertias, 'bx-')
plt.xlabel('k (Number of Clusters)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True, alpha=0.3)


# Add elbow point annotation
# Calculate the angle of the elbow
angles = []
for i in range(1, len(inertias)-1):
    v1 = [1, inertias[i] - inertias[i-1]]
    v2 = [1, inertias[i+1] - inertias[i]]
    angle = np.abs(np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0]))
    angles.append(angle)

elbow_point = np.argmax(angles) + 2  # Add 2 because we started from k=1
plt.annotate(f'Elbow Point (k={elbow_point})',
            xy=(elbow_point, inertias[elbow_point-1]),
            xytext=(elbow_point+1, inertias[elbow_point-1]+1000),
            arrowprops=dict(facecolor='red', shrink=0.05))
plt.show()

# 3. Calculate silhouette scores
from sklearn.metrics import silhouette_score

silhouette_scores = []
K = range(2, 11)  # Silhouette score needs at least 2 clusters

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(K, silhouette_scores, 'rx-')
plt.xlabel('k (Number of Clusters)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.grid(True, alpha=0.3)

# Add best k annotation
best_k = K[np.argmax(silhouette_scores)]
plt.annotate(f'Best k={best_k}',
            xy=(best_k, max(silhouette_scores)),
            xytext=(best_k+1, max(silhouette_scores)-0.01),
            arrowprops=dict(facecolor='red', shrink=0.05))
plt.show()

print(f"Based on the elbow method, the optimal number of clusters appears to be {elbow_point}")
print(f"Based on silhouette analysis, the optimal number of clusters is {best_k}")

# Print cluster size distribution for the recommended number of clusters
kmeans_optimal = KMeans(n_clusters=elbow_point, random_state=42)
cluster_labels = kmeans_optimal.fit_predict(X_scaled)
cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
print("\nCluster size distribution:")
for cluster, size in cluster_sizes.items():
    print(f"Cluster {cluster}: {size} samples")

# STEP 2 - With your chosen number of clusters k, find the k-means clustering your data. Plot clustering result with different colors. For example, cluster 1 in red, cluster 2 in blue, etc. You can repeat with different k to find the better number of clusters.

# Determine the optimal number of clusters
optimal_k_elbow = elbow_point
optimal_k = optimal_k_elbow
print(f"Using the recommended number of clusters: {optimal_k}")

# Perform K-Means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Plot the clustering results
plt.figure(figsize=(10, 6))
plt.scatter(X['pts_home'], X['fg_pct_home'], c=cluster_labels, cmap='viridis', alpha=0.6)
plt.xlabel('Points Scored (Home)')
plt.ylabel('Field Goal Percentage (Home)')
plt.title(f'K-Means Clustering (k={optimal_k})')
plt.colorbar(label='Cluster')
plt.grid(True, alpha=0.3)
plt.show()

# Visualize cluster centroids
centroids = kmeans.cluster_centers_
plt.figure(figsize=(10, 6))
plt.scatter(X['pts_home'], X['fg_pct_home'], c=cluster_labels, cmap='viridis', alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, c='red', label='Centroids')
plt.xlabel('Points Scored (Home)')
plt.ylabel('Field Goal Percentage (Home)')
plt.title(f'K-Means Clustering Centroids (k={optimal_k})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# STEP 3 - Calculate the centroid (mean of x axis, mean of y axis) of each cluster and radii that covers 90 % of data of the cluster. 

# Calculate cluster centroid
centroids = kmeans.cluster_centers_
print("Cluster Centroids:")
for i, centroid in enumerate(centroids):
    print(f"Cluster {i}: ({centroid[0]:.2f}, {centroid[1]:.2f})")

# Calculate cluster radii
cluster_radii = []
for i in range(optimal_k):
    cluster_data = X[cluster_labels == i]
    distances = np.sqrt((cluster_data['pts_home'] - centroids[i, 0])**2 + (cluster_data['fg_pct_home'] - centroids[i, 1])**2)
    radius = np.percentile(distances, 90)
    cluster_radii.append(radius)

print("\nCluster Radii (covering 90% of data):")
for i, radius in enumerate(cluster_radii):
    print(f"Cluster {i}: {radius:.2f}")

# **Step 4.** Plot the circles centered at the centroid with radius calculated in step 3 on top of the plot from step 2. Mark the centroid with ‘X’.

# Plot the clustering results with centroids and radii
plt.figure(figsize=(10, 6))
plt.scatter(X['pts_home'], X['fg_pct_home'], c=cluster_labels, cmap='viridis', alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, c='red', label='Centroids')

for i, (centroid, radius) in enumerate(zip(centroids, cluster_radii)):
    circle = plt.Circle(centroid, radius, fill=False, color='r', linewidth=2)
    plt.gca().add_patch(circle)
    plt.text(centroid[0] + 0.5, centroid[1] + 0.5, f"Cluster {i}", color='red')

plt.xlabel('Points Scored (Home)')
plt.ylabel('Field Goal Percentage (Home)')
plt.title(f'K-Means Clustering (k={optimal_k})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# **Step 5.** In a markdown cell, discuss the best k
optimal_k = optimal_k_elbow
print(f"Using the recommended number of clusters: {optimal_k}")


# Step 6
from sklearn.mixture import GaussianMixture
optimal_k = 4  # Based on the elbow method from previous analysis

# Get the scaled data
X = range_dataframe[['pts_home', 'fg_pct_home']].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

gmm = GaussianMixture(n_components=optimal_k, random_state=42)
cluster_labels = gmm.fit_predict(X_scaled)

# Plot the clustering results
plt.figure(figsize=(10, 6))
plt.scatter(X['pts_home'], X['fg_pct_home'], c=cluster_labels, cmap='viridis', alpha=0.6)
plt.xlabel('Points Scored (Home)')
plt.ylabel('Field Goal Percentage (Home)')
plt.title(f'GMM Clustering (k={optimal_k})')
plt.colorbar(label='Cluster')
plt.grid(True, alpha=0.3)
plt.show()

# Transform means back to original scale
centroids = scaler.inverse_transform(gmm.means_)
covariances = []

# Transform covariances back to original scale
for cov_matrix in gmm.covariances_:
    # For full covariance matrices, need to transform back using the scaling factors
    scale_factors = np.diag(scaler.scale_)
    transformed_cov = scale_factors @ cov_matrix @ scale_factors.T
    covariances.append(transformed_cov)

# Print centroids and covariance matrices
print("\nCluster Centroids and Covariance Matrices:")
for i in range(optimal_k):
    print(f"\nCluster {i}:")
    print(f"Centroid: ({centroids[i][0]:.2f}, {centroids[i][1]:.2f})")
    print("Covariance Matrix:")
    print(covariances[i])

plt.figure(figsize=(10, 6))
plt.scatter(X['pts_home'], X['fg_pct_home'], c=cluster_labels, cmap='viridis', alpha=0.6)

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, c='red', label='Centroids')

# Plot confidence ellipses (90% confidence interval)
for i in range(optimal_k):
    # Calculate eigenvalues and eigenvectors of covariance matrix
    eigenvals, eigenvecs = np.linalg.eigh(covariances[i])
    
    # Calculate angle of ellipse
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    
    # Calculate chi-square value for 90% confidence interval
    chi2_val = stats.chi2.ppf(0.9, df=2)
    
    # Create and plot ellipse
    ell = plt.matplotlib.patches.Ellipse(
        xy=centroids[i],
        width=2 * np.sqrt(chi2_val * eigenvals[0]),
        height=2 * np.sqrt(chi2_val * eigenvals[1]),
        angle=angle,
        fill=False,
        color='red'
    )
    plt.gca().add_patch(ell)
    plt.text(centroids[i][0] + 0.5, centroids[i][1] + 0.005, f"Cluster {i}", color='red')

plt.xlabel('Points Scored (Home)')
plt.ylabel('Field Goal Percentage (Home)')
plt.title(f'GMM Clustering with Confidence Ellipses (k={optimal_k})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\nGMM Clustering Analysis:")
print(f"Number of clusters: {optimal_k}")

# Calculate cluster sizes
cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
print("\nCluster size distribution:")
for cluster, size in cluster_sizes.items():
    print(f"Cluster {cluster}: {size} samples ({size/len(cluster_labels)*100:.1f}%)")

# Calculate BIC and AIC scores
print(f"\nBIC Score: {gmm.bic(X_scaled):.2f}")
print(f"AIC Score: {gmm.aic(X_scaled):.2f}")

# Print average probabilities for each cluster
print("\nAverage probability of cluster membership:")
probabilities = gmm.predict_proba(X_scaled)
for i in range(optimal_k):
    print(f"Cluster {i}: {probabilities[:, i].mean():.3f}")

# **Step 7. (extra credit)** Plot 2D Gaussian curve calculated in step 6 as shown in the 2D GMM with EM examples in the lecture slides.
from mpl_toolkits.mplot3d import Axes3D

# Create a grid of points
x = np.linspace(X['pts_home'].min(), X['pts_home'].max(), 100)
y = np.linspace(X['fg_pct_home'].min(), X['fg_pct_home'].max(), 100)
X_grid, Y_grid = np.meshgrid(x, y)
pos = np.dstack((X_grid, Y_grid))

# Calculate the Gaussian mixture probability density
Z = np.zeros_like(X_grid)
for i in range(optimal_k):
    # Get the parameters for this component
    weight = gmm.weights_[i]
    mean = centroids[i]
    cov = covariances[i]
    
    # Create multivariate normal distribution
    rv = stats.multivariate_normal(mean, cov)
    
    # Add weighted probability density to total
    Z += weight * rv.pdf(pos)

# Create 3D surface plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(X_grid, Y_grid, Z, cmap='viridis', alpha=0.8)

# Add color bar
plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Set labels and title
ax.set_xlabel('Points Scored (Home)')
ax.set_ylabel('Field Goal Percentage (Home)')
ax.set_zlabel('Probability Density')
ax.set_title('2D Gaussian Mixture Model Surface Plot')

# Adjust viewing angle for better visualization
ax.view_init(elev=20, azim=45)

plt.show()

# Create contour plot
plt.figure(figsize=(10, 8))
plt.contour(X_grid, Y_grid, Z, levels=20, cmap='viridis')
plt.scatter(X['pts_home'], X['fg_pct_home'], c='red', alpha=0.1, s=1)
plt.colorbar(label='Probability Density')
plt.xlabel('Points Scored (Home)')
plt.ylabel('Field Goal PercentagAe (Home)')
plt.title('GMM Contour Plot with Data Points')
plt.show()

