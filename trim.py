import pandas as pd

# Load
df = pd.read_excel('onlineRetail.xlsx')

# Keep only the first 2000 rows
df = df.iloc[:2000]

# Save the trimmed data back to Excel
df.to_excel('onlineRetailTRIM.xlsx', index=False)


print(f"Trimmed file saved to ")
