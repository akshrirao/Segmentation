import pandas as pd
import numpy as np  
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import streamlit as st

st.set_page_config(page_title="Customer Segmentation", page_icon=":bar_chart:", layout="wide")

#load the model
kmeans=joblib.load('kmeans_model.pkl')
scaler=joblib.load('scaler.pkl')

#title and description
st.title("Customer Segmentation Dashboard")
st.text('''
Customer Segmentation Model
K-means algorithm based
        - Annual income
        - Age
        - Spending Score
''')

#Side bar
st.sidebar.header("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Upload A CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df=pd.read_csv(r"Mall_Customers.csv")    

    st.write("Data Preview:")
    st.dataframe(df.head())
    
    #feature selection
    X=df[['Age','Annual Income (k$)','Spending Score (1-100)']]
    
    #scaling
    X_scaled= scaler.transform(X)
    
    #clusters
    clusters=kmeans.predict(X_scaled)
    df['cluster'] = clusters
    
    #PCA visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame()
    pca_df['PCA1'] = X_pca[:, 0]
    pca_df['PCA2'] = X_pca[:, 1]
    pca_df['cluster'] = clusters.astype(str)    

    st.dataframe(pca_df.head(2))

    #pca cluster visualization
    plt.figure(figsize=(10,3))  
    sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='cluster', palette='Set1', s=100)
    plt.title('Customer Segments (PCA Visualization)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.legend(title='Cluster')
    st.pyplot(plt)  

    # cluster analysis
    st.subheader("Cluster Analysis")
    cluster_summary = df.groupby('cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().reset_index()
    st.dataframe(cluster_summary)   


    #real time prediction
    st.subheader("Predict Cluster for New Customer")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
    with col2:
        annual_income = st.number_input("Annual Income (k$)", min_value=1, max_value=200, value=50)
    with col3:
        spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)
    
    # prediction button
    if st.button("Predict Cluster"):
        new_data = np.array([[age, annual_income, spending_score]])
        new_data_scaled = scaler.transform(new_data)
        predicted_cluster = kmeans.predict(new_data_scaled)[0]
        st.write(f"The predicted cluster for the new customer is: Cluster {predicted_cluster}")

    # download data 
    st.subheader("Download Segmented Data")
    if st.button("Download CSV"):
        df.to_csv("segmented_customers.csv", index=False)
        st.write("Data downloaded successfully!")   
        
