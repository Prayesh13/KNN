import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.datasets import make_classification
from mlxtend.plotting import plot_decision_regions

X,y = make_classification(n_samples=300, n_features=2,n_informative=1,n_redundant=0,n_clusters_per_class=1,random_state=0)
X = X + 1.5*np.random.randn(X.shape[0],X.shape[1])
st.sidebar.title("KNN Classifier Hyperparameter Tuning")


k_neighbors = st.sidebar.slider("Number of Neighbors (K) : ",1,15,5)
weights = st.sidebar.selectbox("Weight Function ",['uniform','distance'])

algorithm = st.sidebar.selectbox("Algorithm ",['auto', 'ball_tree', 'kd_tree', 'brute'])

p = st.sidebar.slider("Power parameter for the Minkowski metric : ",1,10,2)

leaf_size = st.sidebar.slider("leaf_size ",10,100,30)

btn = st.sidebar.button("Predict")

if btn:
    knn_model = KNeighborsClassifier(weights=weights, algorithm=algorithm, p=p, leaf_size=leaf_size)
    knn_model.fit(X,y)
    y_pred = knn_model.predict(X)

    st.write("Accuracy Score : "+str(accuracy_score(y,y_pred)))
    fig, ax = plt.subplots()

    plot_decision_regions(X,y,clf=knn_model)

    ax.set_title("KNN Classification Visulization")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()

    st.pyplot(fig)