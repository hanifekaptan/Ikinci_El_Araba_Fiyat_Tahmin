#!/usr/bin/env python
# coding: utf-8

# # Car Prediction #
# İkinci el araç fiyatlarını (özelliklerine göre) tahmin eden modeller oluşturma ve MLOPs ile Hugging Face üzerinden yayımlayacağız.

import pandas as pd
# from sklearn.model_selection import train_test_split #veri setini bölme işlemleri
from sklearn.linear_model import LinearRegression #Doğrusal regresyon
# from sklearn.metrics import r2_score,mean_squared_error #modelimizin performansını ölçmek için
from sklearn.compose import ColumnTransformer #Sütun dönüşüm işlemleri
from sklearn.preprocessing import OneHotEncoder, StandardScaler # kategori - sayısal dönüşüm  ve ölçeklendirme
from sklearn.pipeline import Pipeline #Veri işleme hattı

df=pd.read_excel('cars.xls')

X = df.drop("Price", axis=1) #bağımsız değişkenler
y = df["Price"] #bağımlı değişken, tahmin edilecek değer

# Veri ön işleme, standartlaştırma ve OHE işlemlerini otomatikleştiriyoruz (standarlaştırıyoruz).
# Artık preprocess kullanarak kullanıcında arayüz aracılığıyla gelen veriyi modelimize uygun hale çevirebiliriz.
preprocess=ColumnTransformer(
    transformers=[('num',StandardScaler(),['Mileage', 'Cylinder','Liter','Doors']),
        ('cat',OneHotEncoder(),['Make','Model','Trim','Type'])])

my_model=LinearRegression()

#pipeline ı tanımla
pipe=Pipeline(steps=[('preprocessor',preprocess),('model',my_model)]) # bu işlem ile web sitesinden alacağımız verilerin dönüşümü sağlanır.

# İyi bir tahmin sonucu verdiği için bütün veri setini eğitim için kullanacağız.
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# pipe.fit(X_train,y_train)  
# y_pred=pipe.predict(X_test)
# print('RMSE',mean_squared_error(y_test,y_pred)**0.5)
# print('R2',r2_score(y_test,y_pred))

pipe.fit(X,y)


# Streamlit ile modeli yayma/deploy/kullanıma sunma
# Python ile yapılan çalışmnalrın hızlı bir şekilde deploy edilmesi için HTML render arayüzler tasarlamanızı sağlar.

import streamlit as st
#price tahmin fonksiyonu tanımla
def price(mileage, make, model, trim, type, cylinder, liter, doors, cruise, sound, leather):
    input_data= pd.DataFrame({"Mileage":[mileage],
                              "Make":[make],
                              "Model":[model],
                              "Trim":[trim],
                              "Type":[type],
                              "Cylinder":[cylinder],
                              "Liter":[liter],
                              "Doors":[doors],
                              "Cruise":[cruise],
                              "Sound":[sound],
                              "Leather":[leather]})
    prediction = pipe.predict(input_data)[0]
    return prediction

st.title("2. El Araba Fiyatı Tahmin:red_car: @hanifekaptan")
st.write('Arabanın özelliklerini seçiniz')
mileage=st.number_input('Kilometre',100,200000)
make=st.selectbox('Marka',df['Make'].unique())
model=st.selectbox('Model',df[df['Make']==make]['Model'].unique())
trim=st.selectbox('Trim',df[(df['Make']==make) &(df['Model']==model)]['Trim'].unique())
car_type=st.selectbox('Araç Tipi',df[(df['Make']==make) &(df['Model']==model)&(df['Trim']==trim)]['Type'].unique())
cylinder=st.selectbox('Cylinder',df['Cylinder'].unique())
liter=st.number_input('Yakıt Hacmi',1,10)
doors=st.selectbox('Kapı Sayısı',df['Doors'].unique())
cruise=st.radio('Hız Sbt.',[True,False])
sound=st.radio('Ses Sistemi',[True,False])
leather=st.radio('Deri Döşeme.',[True,False])
if st.button('Tahmin'):
    pred=price(mileage, make, model, trim, car_type, cylinder, liter, doors, cruise, sound, leather)
    st.write('Fiyat ($):', round(pred,2))
