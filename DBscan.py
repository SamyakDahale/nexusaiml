import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

# Load
df = pd.read_excel('preprocessed_online_retail.xlsx')

 # Data standardization
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.select_dtypes(include=[np.number]))

# Applying DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust eps and min_samples as needed
dbscan.fit(scaled_data)

# Add cluster labels to   data
df['Cluster'] = dbscan.labels_

# Analyze  resultsof custering
print("Cluster value counts:")
print(df['Cluster'].value_counts())

# Visualizing clusters in 2 dimnsions
if scaled_data.shape[1] == 2:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=scaled_data[:, 0], y=scaled_data[:, 1], hue=df['Cluster'], palette='Set1', legend='full')
    plt.title('DBSCAN Clustering')
    plt.show()
else:
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(scaled_data)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=reduced_data[:, 0], y=reduced_data[:, 1], hue=df['Cluster'], palette='Set1', legend='full')
    plt.title('DBSCAN Clustering (PCA Reduced)')
    plt.show()

# Save in new file
df.to_excel('DBSCAN_anomaly_detection_results.xlsx', index=False)
print(f"Clustered data saved to 'DBSCAN_anomaly_detection_results.xlsx'.")