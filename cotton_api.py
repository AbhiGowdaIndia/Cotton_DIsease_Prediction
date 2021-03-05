from tensorflow.keras.models import load_model
import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from PIL import Image,ImageOps

fin_model=load_model('Model-3.h5')

def model_predict(img_path):
    size=(224,224)
    img = ImageOps.fit(img_path,size,Image.ANTIALIAS)
    #img = image.load_img(imag, target_size=(224, 224))
    # Preprocessing the image
    x = image.img_to_array(img)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
    img_data=preprocess_input(x)
    pred=fin_model.predict(img_data)
    preds=np.argmax(pred, axis=1)
    if preds==0:
        preds="The leaf is diseased cotton leaf"
    elif preds==1:
        preds="The leaf is diseased cotton plant"
    elif preds==2:
        preds="The leaf is fresh cotton leaf"
    else:
        preds="The leaf is fresh cotton plant"
    return preds


def main():
	st.title('Cotton Disease Prediction using Deep Learning')
	st.write('Upload an image of cotton plant or leaf to check whether it is a diseased cotton leaf or not.')

	html_temp="""
    <div style="background-color:tomato;padding:10px;">
    <h2 style="color:white;text-align:center;">Cotton Disease Prediction</h2>
    </div>
    """
	st.markdown(html_temp,unsafe_allow_html=True)
	
	st.subheader("Upload Image")
	image_p= st.file_uploader("", type=['png','jpeg','jpg'])
	if image_p is not None:
		img=Image.open(image_p)
		st.image(img,width=240,height=240)
		if st.button("Predict"):
			result=model_predict(img)
			st.success('Result : "{}"'.format(result))
		
if __name__=='__main__':
	main()
