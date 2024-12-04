from sklearn.metrics import calinski_harabasz_score
import pandas as pd

# Load
df = pd.read_excel('anomaly_detection_results.xlsx')
X = df[['Quantity', 'Price', 'TotalAmount', 'PurchaseFrequency',
        'AvgTransactionAmount', 'ProductDiversity', 'CountryEncoded']]

#  Assign binary labels (Anomaly = 1, Normal = 0)
df['Predicted_Label'] = df['Anomaly'].apply(lambda x: 1 if x == -1 else 0)

#  Calculate Index
# Higher values indicate better-defined clusters
ch_score = calinski_harabasz_score(X, df['Predicted_Label'])

# SPrint
print(f"Calinski-Harabasz Index: {ch_score:.4f}")