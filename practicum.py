import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import log_loss, roc_auc_score, confusion_matrix, classification_report
import numpy as np;
from sklearn.preprocessing import StandardScaler


data_regression = pd.read_csv('housing23.csv')

data_classification = pd.read_csv("smoke2.csv")





def load_models():
    model_classification = pickle.load(open('Ridge.pkl', 'rb'))
    model_regression = pickle.load(open('TweedieRegressor.pkl', 'rb'))
    return model_classification, model_regression

def machine_learning():
    page=st.sidebar.selectbox("Выберите страницу",['Датасеты','Предикт'])
    if (page=="Датасеты"):
        "Housing.csv:"
        "price : Цена недвижимости в долларах или валюте, которая представлена в датасете."
        "area: Площадь недвижимости в квадратных футах или метрах квадратных."
        "bedrooms: Количество спален в недвижимости."
        "bathrooms: Количество ванных комнат в недвижимости."
        "stories: Количество этажей в здании недвижимости."
        "mainroad: Присутствие (yes) или отсутствие (no) основной дороги рядом с недвижимостью."
        "guestroom: Наличие (yes) или отсутствие (no) гостевой комнаты в недвижимости."
        "basement: Наличие (yes) или отсутствие (no) подвала в недвижимости."
        "hotwaterheating: Наличие (yes) или отсутствие (no) системы горячего водоснабжения."
        "airconditioning: Наличие (yes) или отсутствие (no) системы кондиционирования воздуха."
        "parking: Количество парковочных мест, доступных недвижимости."
        "prefarea: Наличие (yes) или отсутствие (no) приоритетного района (предпочтительной локации) для недвижимости."
        "furnishingstatus: Статус обстановки недвижимости, который может быть 'furnished' (меблированная), 'semi-furnished' (частично меблированная) или 'unfurnished' (немеблированная)."
        "Smoke_detector_tast.csv:"
        "UTC: Временная метка (в формате Unix time) для измерений."
        "Temperature[C]: Температура в градусах Цельсия."
        "Humidity[%]: Влажность в процентах."
        "TVOC[ppb]: Концентрация летучих органических соединений в частях на миллиард."
        "eCO2[ppm]: Концентрация углекислого газа в частях на миллион."
        "Raw H2: Сырые данные о концентрации водорода."
        "Raw Ethanol: Сырые данные о концентрации этанола."
        "Pressure[hPa]: Атмосферное давление в гектопаскалях."
        "PM1.0: Концентрация частиц диаметром менее 1.0 микрона."
        "PM2.5: Концентрация частиц диаметром менее 2.5 микрона."
        "NC0.5: Количество частиц диаметром более 0.5 микрона."
        "NC1.0: Количество частиц диаметром более 1.0 микрона."
        "NC2.5: Количество частиц диаметром более 2.5 микрона."
        "CNT: Счетчик, предположительно, относящийся к количеству."
        "Fire Alarm: Статус сигнала пожарной тревоги ('Yes' - да, сработала тревога; 'No' - нет, тревога не сработала)."
    elif (page=="Предикт"):
        st.title("Datasets")
        uploaded_file = st.file_uploader("Загрузите ваш CSV файл", type="csv")
        if uploaded_file is not None:
            if uploaded_file.name == "smoke2.csv":
                st.write("Файл smoke2.csv был загружен")
                y = data_classification["Fire Alarm_Yes"]
                X = data_classification.drop(["Fire Alarm_Yes","UTC"], axis=1)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)
                model_classification = pickle.load(open('Ridge.pkl', 'rb'))
                predictions_classification = model_classification.predict(X_test)
                accuracy_classification = accuracy_score(y_test, predictions_classification)
                st.success(f"Точность: {accuracy_classification}")

            elif uploaded_file.name == "housing23.csv":
                st.write("Файл housing23.csv был загружен")
                model_regression = pickle.load(open('TweedieRegressor.pkl', 'rb'))
                data_regression = pd.read_csv('C:/Users/ZALMAN/Downloads/housing23.csv')
                y = data_regression["price"]
                X = data_regression.drop(["price"], axis=1)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
                predictions_regression = model_regression.predict(X_test)
                r2_score_regression = r2_score(y_test, predictions_regression)
                st.success(f"Коэффициент детерминации (R²): {r2_score_regression}")


            else:
                st.write("Загружен файл неизвестного формата")

if __name__ == "__main__":
    machine_learning()