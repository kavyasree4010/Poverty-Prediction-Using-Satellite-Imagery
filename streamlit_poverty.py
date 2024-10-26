import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf 


def load_model():
    model = tf.keras.models.load_model('poverty_classification_model.h5')  # Modify this line based on your model
    return model

model = load_model()


def model_predict(img, model):
    img = img.resize((256,256))  
    img = np.array(img)
    img = np.expand_dims(img, axis=0) 
    img=np.delete(img,0,1) 
    img=np.resize(img,(256,256,3)) 
    img=np.array([img])/255 
    pred=model.predict(img) 
    return "Poor" if pred<=0.5 else "Rich" 

st.title("Poverty Classification Model")

uploaded_file = st.file_uploader("Choose the Satellite Image of that Place", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image,width=200)
    st.write("")
    st.write("Here is the Uploaded Image")

    if st.button('Classify'):
        predictions = model_predict(image, model)
        st.write(predictions)
