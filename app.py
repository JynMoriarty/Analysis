import pandas as pd
import streamlit as st
from PIL import Image
import pickle
image = Image.open('bellevue.jpg')
import numpy as np
import pydeck as pdk
p1 = open('modelRid.pkl', 'rb') 
ridge_model = pickle.load(p1)
p2 = open("modellr.pkl","rb")
lr_model = pickle.load(p2)
p3 = open("modelLasso.pkl","rb")
lasso_model = pickle.load(p3)
p4= open("modelEN.pkl","rb")
elastic_model = pickle.load(p4)
st.title("Indicateur de prix d'une maison dans le comté de King(Washington")
st.image(image, caption='Bellevue,City in King County')

st.header("Entrez les caractéristiques de votre maison")
zipcode = [98001, 98002, 98003, 98004, 98005, 98006, 98007, 98008, 98010,
       98011, 98014, 98019, 98022, 98023, 98024, 98027, 98028, 98029,
       98030, 98031, 98032, 98033, 98034, 98038, 98039, 98040, 98042,
       98045, 98052, 98053, 98055, 98056, 98058, 98059, 98065, 98070,
       98072, 98074, 98075, 98077, 98092, 98102, 98103, 98105, 98106,
       98107, 98108, 98109, 98112, 98115, 98116, 98117, 98118, 98119,
       98122, 98125, 98126, 98133, 98136, 98144, 98146, 98148, 98155,
       98166, 98168, 98177, 98178, 98188, 98198, 98199]
    
    
regression=st.sidebar.selectbox("Choissisez l'algorithme de régression",("Linear","Ridge","Lasso","ElasticNet"))

bedrooms = st.number_input('Chambres :', min_value=1, value=1,step=1)
bathrooms = st.number_input('Salle de bains',min_value=1.0,value=1.0,step=0.25)
sqm_living =st.number_input("Surface habitable (en m2) : ",min_value =1,value=100,step=1)
sqm_lot = st.number_input ("Surface du Terrain (en m2) : ",min_value =1,value=200,step=1)
floors =st.number_input("Nombre d'étages : ,",min_value=0,value=1,step=1)
waterfront =st.selectbox("Est ce qu'il y'a une vue sur la mer ?",["Oui","Non"])
if waterfront == "Oui":
    waterfront = 1
elif waterfront == "Non":
    waterfront = 0
view = st.slider("Qualité de la vue : ",0,4,1)	
condition = st.slider("Condition de la Maison",0,5,1)	
grade=st.slider("Design de la maison  ",0,13,7)
sqm_above  = st.number_input ("Surface intérieur au dessus du sol (en m2) : ",min_value =1,value=50,step=1)
sqm_basement = st.number_input ("Surface de la cave (en m2) : ",min_value =0,value=50,step=1)
zipcode = st.selectbox("Choissisez le code postal de votre maison (format :98XXX): ",zipcode)
sqm_living15  =st.number_input("Surface habitable moyen local (en m2 : ",min_value =1,value=120,step=1)
sqm_lot15 = st.number_input ("Surface du Terrain moyen local (en m2) : ",min_value =1,value=200,step=1)
house_age =st.selectbox("Age de la maison (0:(0-20),1:(20-40),2:(40-60),3:(60-80),4:(80-100),5:(100-115)",[0,1,2,3,4,5])
renovated = st.selectbox("Est-ce que votre maison a été rénové ? ",["Oui","Non"])
if renovated == "Oui":
        renovated =1
elif renovated == "Non":
        renovated =0

liste = [bedrooms,bathrooms,sqm_living,sqm_lot,floors,int(waterfront),view,condition,grade,sqm_above,sqm_basement,zipcode,sqm_living15,sqm_lot15,house_age,int(renovated)]
columns=["bedrooms","bathrooms","sqm_living","sqm_lot","floors","waterfront","view","condition","grade","sqm_above","sqm_basement","zipcode","sqm_living15","sqm_lot15","house_age","renovated"]
prediction = pd.DataFrame(np.array(liste).reshape(1,-1),columns=columns)
if st.button("Prédire"):
    if regression == "Linear":
        
        price = int(lr_model.predict(prediction))
        st.success("{}$".format(price))
    elif regression == "Ridge":
        price = int(ridge_model.predict(prediction))
        st.success("{}$".format(price))
    elif regression == "Lasso":
        price = int(lasso_model.predict(prediction))
        st.success("{}$".format(price))
    else:
        price = int(elastic_model.predict(prediction))
        st.success("{}$".format(price))
    

 