import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Load dataset
df = pd.read_excel('onlineRetailTRIM.xlsx')

# Drop missng & duplicate value
df = df.dropna(subset=['Customer ID', 'Description'])
df = df.drop_duplicates()

df = df[(df['Quantity'] > 0) & (df['Price'] > 0)]

# Standardize
df['Description'] = df['Description'].str.strip().str.lower()


# Create a TotalAmount column
df['TotalAmount'] = df['Quantity'] * df['Price']

# Extract time-based
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%Y %H:%M')
df['Year'] = df['InvoiceDate'].dt.year
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day
df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
df['Hour'] = df['InvoiceDate'].dt.hour


# Create customer behavior metrics
customer_behavior = df.groupby('Customer ID').agg({
    'Invoice': 'count',
    'TotalAmount': 'mean',
    'StockCode': 'nunique'
}).reset_index()

customer_behavior.columns = ['Customer ID', 'PurchaseFrequency', 'AvgTransactionAmount', 'ProductDiversity']

df = df.merge(customer_behavior, on='Customer ID', how='left')

# Encode 'Country' with label encoding
le = LabelEncoder()
df['CountryEncoded'] = le.fit_transform(df['Country'])

#  apply scaling --
scaler = MinMaxScaler()
df[['Quantity', 'Price', 'TotalAmount', 'PurchaseFrequency', 'AvgTransactionAmount', 'ProductDiversity']] = \
    scaler.fit_transform(df[['Quantity', 'Price', 'TotalAmount', 'PurchaseFrequency', 'AvgTransactionAmount', 'ProductDiversity']])

# Drop unecessary datasets
df_final = df.drop(columns=['Invoice', 'InvoiceDate', 'Customer ID', 'Country', 'Description', 'Year', 'Month', 'Day'])

# Save preprocessed data to a new XLSX
df_final.to_excel('preprocessed_online_retail.xlsx', index=False)
print("Preprocessing complete. Preprocessed data saved to 'preprocessed_online_retail.xlsx'.")