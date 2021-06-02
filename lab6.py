import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

# models = ['KNN5', 'DecisionTree', 'LogRegr', 'SVC']
models = ['KNN', 'DecisionTree', 'LogRegr']
models_dict = {'KNN': KNeighborsClassifier(),
               'DecisionTree': DecisionTreeClassifier(random_state=42),
               'LogRegr': LogisticRegression(random_state=42)}


@st.cache
def load_data():
    data = pd.read_csv('C:\\Users\\Дасупс\\Downloads\\heart.csv', sep=",")
    return data


def data_preprocessing(in_data):
    out_data = in_data.copy()
    columns_to_scale = ['chol', 'trestbps', 'thalach', 'age']
    scaler = MinMaxScaler()
    # масшабирование колонок
    for col in columns_to_scale:
        out_data[[col]] = scaler.fit_transform(out_data[[col]])
    # разделение данных на признаки и целевую переменную
    X = out_data.drop('target', axis=1)
    y = out_data.target
    return train_test_split(X,y,test_size=0.2,random_state=42)


def print_results(estimator, data):
    X_train, X_test, y_train, y_test = data_preprocessing(data)
    estimator.fit(X_train, y_train)
    y_predicted = estimator.predict(X_test)
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted)
    proba = estimator.predict_proba(X_test)
    roc = roc_auc_score(y_test, proba[:,1])
    st.subheader(f'Accuracy = {round(accuracy,3)}')
    st.subheader(f'Precision = {round(precision, 3)}')
    st.subheader(f'ROC AUC = {round(roc, 3)}')
    fig1, ax = plt.subplots()
    plot_confusion_matrix(estimator, X_test, y_test, ax=ax, cmap=plt.cm.Blues)
    st.pyplot(fig1)



st.sidebar.header('Сравнение моделей машинного обучения')
models_select = st.sidebar.multiselect('Выберите модель/модели:', models)
data = load_data()
st.header('Оценка качества моделей:')

if 'LogRegr' in models_select:
    st.subheader('Логистическая регрессия')
    log_reg_l1 = st.slider('l1-ratio * 10', min_value=0, max_value=10, value=5, step=1)
    if st.checkbox('Описание гиперпараметра'):
        '''
        Гиперпараметр l1-ratio относится к смешанной регуляризации elasticnet
        и определяет, насколько сильным будет влияние l1 и l2.
        l1-ratio = 0 полностью соответствует регуляризации Тихонова l2,
        l1-ratio = 1 - LASSO-регуляризации.
        Поскольку слайдер в streamlit принимает только целочисленные значения, 
        значение параметра задаётся  умноженным на 10.
        '''
    log_model = LogisticRegression(penalty='elasticnet', l1_ratio=log_reg_l1 * 0.1, solver='saga')
    print_results(log_model, data)
if 'DecisionTree' in models_select:
    st.subheader('Решающее дерево')
    max_depth = st.slider('max_depth:', min_value=1, max_value=10, value=5, step=1)
    tree_model = DecisionTreeClassifier(max_depth=max_depth)
    print_results(tree_model, data)
if 'KNN' in models_select:
    st.subheader('K ближайших соседей')
    knn_slider = st.slider('n_neighbors', min_value=1, max_value=int(data.shape[0] * 0.8) -1, value=5, step=1)
    knn_model = KNeighborsClassifier(n_neighbors=knn_slider)
    print_results(knn_model, data)