import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
import streamlit as st

st.set_page_config(
    page_title="KNN Visualizer App",
    page_icon="👋",
)

st.write("# Welcome! 👋")

st.sidebar.title("KNN Regressor Visulization")

# samples
n_sample = 200
x = np.linspace(-3,3,n_sample).reshape(-1,1)
y = np.sin(x).ravel() + 0.5*np.random.randn(n_sample)
print(y)

# sidebar slider
k_neighbors = st.sidebar.slider("Number of Neighbors (K) : ",1,15,5)

# KNN Model :
knn_model = KNeighborsRegressor(n_neighbors=k_neighbors)

knn_model.fit(x,y)

y_pred = knn_model.predict(x)

btn = st.sidebar.button("predict")

# Plotting the graphs
fig,ax = plt.subplots()
ax.scatter(x,y,color='blue',label='Data Points')
if btn:
    ax.plot(x,y_pred,color='red',label='Predicted Line')
    st.write("R2 Score : " + str(r2_score(y, y_pred)))
ax.set_title("KNN Regression Visulization")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.legend()



st.pyplot(fig)