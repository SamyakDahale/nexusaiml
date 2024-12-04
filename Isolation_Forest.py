import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Load
df = pd.read_excel('preprocessed_online_retail.xlsx')

# Select the relevant features for anomaly detection (numerical ones)
X = df[['Quantity', 'Price', 'TotalAmount', 'PurchaseFrequency',
        'AvgTransactionAmount', 'ProductDiversity', 'CountryEncoded']]

# Configure the Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)  # Adjust contamination as per your need
df['Anomaly'] = model.fit_predict(X)


# In Isolation Forest, -1 means anomaly, and 1 means normal
df['Anomaly_Label'] = df['Anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')

#  Visualize the anomalies ; use a 2D plot (PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Anomaly'], cmap='coolwarm', label='Normal vs Anomaly')
plt.title('Anomaly Detection using Isolation Forest')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()

# Save in new file
df.to_excel('anomaly_detection_results.xlsx', index=False)
print("Anomaly detection complete. Results saved to 'anomaly_detection_results.xlsx'.")
