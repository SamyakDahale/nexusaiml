from sklearn.metrics import silhouette_score
import pandas as pd

df = pd.read_excel('anomaly_detection_results.xlsx')
X = df[['Quantity', 'Price', 'TotalAmount', 'PurchaseFrequency',
        'AvgTransactionAmount', 'ProductDiversity', 'CountryEncoded']]

# : Assign binary labels (Anomaly = 1, Normal = 0)
df['Predicted_Label'] = df['Anomaly'].apply(lambda x: 1 if x == -1 else 0)

#  Calculate the Silhouette Score
sil_score = silhouette_score(X, df['Predicted_Label'])

#  Print
print(f"Silhouette Score: {sil_score:.4f}")
