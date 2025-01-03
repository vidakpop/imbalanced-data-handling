import streamlit as st
import joblib 
import numpy as np
from PIL import Image
import base64#

#loading the trained model
path='smottomek.pkl'
clf=joblib.load(path)

#defining the feature names 
feature_names=['Length','Diameter','Height','Whole_weight','Shucked_weight','Viscera_weight','Shell_weight','Sex_I','Sex_M']

def get_user_input():
    user_input=[]

    for feature in feature_names:
        value=st.text_input(f'{feature}:','0')
        user_input.append(float(value))
    return np.array(user_input).reshape(1,-1)

#css styling
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html=True)

local_css('style.css')

#Main function to run the app
def main():
    #title and header
    st.title('ABALONE KNOWLEDGE AND PREDICTION APP')
    st.header('Equipping Fishermen and Researchers with Knowledge of Abalone ')

    st.sidebar.title('ABALONE WEBAPP OVERVIEW')
    st.sidebar.subheader('1. What are abalones?')
    st.sidebar.subheader('2. Biology and Habitat')
    st.sidebar.subheader('3. Nutritional Value')
    st.sidebar.subheader('4. Culinary Uses')
    st.sidebar.subheader('5. Predict Abalone Classification')


    #information section about abalones
    st.subheader('What are Abalones?')
    st.write("""
    Abalones are a group of marine gastropod mollusks. They are known for their colorful inner shell and are considered a delicacy in many parts of the world. 
    Abalones have a single rounded to oval shell, with a row of respiratory pores and a flattened, muscular foot. They cling to rocks with this foot and use it to move around.
    """) 
    abalone_img=Image.open('abalone.jpg')
    st.image(abalone_img,caption='Abalone Shell')

    st.subheader('Biology and Habitat')
    st.write("""
    Abalones are found in cold waters, often along the coasts of New Zealand, South Africa, Australia, Japan, and North America. They are herbivores, feeding on algae.
    """)
    habitat_img=Image.open('habitat.jpg')
    st.image(habitat_img,caption='Abalone Habitat')

    st.subheader('Nutritional Value')
    st.write("""
    Abalone is not only a delicacy but also a nutritious food. It is rich in protein, vitamins, and minerals. It contains essential amino acids, omega-3 fatty acids, and is low in fat.
    """)
    nutrition_img=Image.open('nutrition.jpg')
    st.image(nutrition_img,caption='Nutritional Benefits of Abalone')

    st.subheader('Culinary Uses')
    st.write("""
    Abalone can be prepared in various ways, including grilling, steaming, and frying. It is often served as a luxury item in high-end restaurants and is a popular ingredient in Asian cuisine.
    """)
    culinary_img=Image.open('culinary.jpg')
    st.image(culinary_img,caption='Culinary uses of Abalone')

    #user input
    st.subheader('Predict Abalone Classification')
    st.write("Enter the features of the abalone to predict its classification.")

    user_input=get_user_input()

    if st.button('Predict'):
        prediction_proba=clf.predict_proba(user_input)[:,1]
        prediction=clf.predict(user_input)

        st.write(f'Predicted probability of positive class:{prediction_proba[0]:.4f}%')
        st.write(f"Predicted class: {'It is an Abalone' if prediction[0] == 1 else 'not an Abalone'}")

#Run the app
if __name__=='__main__':
    main()
    

