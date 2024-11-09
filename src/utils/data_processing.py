import pandas as pd
from sklearn.decomposition import PCA

# Example usage:
file_path = 'all_vars.txt'  # Path to your file

# Initialize a list to store rows
rows = []

# Open the file and read it line by line
with open(file_path, 'r') as file:
    next(file)  # Skip the first line (header)
    for line in file:
        parts = line.split()
        
        # Extract relevant data: date, name, and value (ignore lat and lon)
        date = parts[0]
        name = parts[1]
        value = float(parts[-1])  # The last column is the value
        
        # Append the extracted data to the rows list
        rows.append([date, name, value])

# Create a DataFrame from the rows
df2 = pd.DataFrame(rows, columns=['date', 'name', 'value'])

# Pivot the DataFrame to make 'name' the columns
df2 = df2.pivot(index='date', columns='name', values='value')

df_physical = df2[[
    'p_actual', 'pr', 'va700', 'va850', 'ta700', 'ta500', 'hurs', 'ua500']
]

# Standardize the data before applying PCA
scaler = StandardScaler()
df_physical_scaled = scaler.fit_transform(df_physical)

# Apply PCA
pca = PCA(n_components=2)  # Adjust the number of components as needed
df_physical_pca = pca.fit_transform(df_physical_scaled)

# Convert the PCA result to a DataFrame for easier manipulation
df_physical_pca = pd.DataFrame(df_physical_pca, columns=['PC1', 'PC2'], index=df_physical.index)

print(df)

# Concatenate the PCA result with the original DataFrame (df) using the date column
df = pd.concat([df, df_physical_pca], axis=1, join='inner')

# print(df)