import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

#Load the dataset
df=pd.read_csv(r"C:\Users\PRIYANKA SHRIRAO\OneDrive\Desktop\DATA SCIENCE_AK\VS CODE\ML Pro-2 Unsupervised learn\Mall_Customers.csv")

#feature selection
X=df[['Age','Annual Income (k$)','Spending Score (1-100)']]

#scalling
scaler=StandardScaler()
X_scaled= scaler.fit_transform(X)

# Train Model
kmeans = KMeans (n_clusters=5, random_state=42 )
kmeans.fit(X_scaled)

# Save Model
import os
os.makedirs('model', exist_ok=True)
joblib.dump(kmeans, 'model/kmeans_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
print("Model trained and saved successfully.")
