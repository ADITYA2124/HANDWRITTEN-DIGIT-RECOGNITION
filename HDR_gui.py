import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from HDR_model import model

# Load your trained model
model = tf.keras.models.load_model('handwritten.model')

def recognize_digit(image):
    try:
        img = np.array(image.convert('L'))  # Convert image to grayscale array
        img = np.invert(img)  # Invert the image
        img = np.array([img])  # Add batch dimension
        prediction = model.predict(img)
        return np.argmax(prediction)
    except Exception as e:
        st.error(f"Error: {e}")
        return None

def main():
    st.title("Handwritten Digit Recognition")
    st.markdown(
        """
        Welcome to the Handwritten Digit Recognition App! This app allows you to upload an image of a handwritten digit 
        (PNG, JPG, or JPEG) and utilizes a pre-trained machine learning model to predict the digit.

        To get started, simply upload an image using the file uploader below. Once uploaded, the app will display the 
        image and provide you with the predicted digit.

        **How it works:**
        - Upload an image of a handwritten digit.
        - The app will resize the image while preserving its aspect ratio.
        - The machine learning model will then predict the digit from the resized image.
        - Finally, the app will display both the uploaded and resized images along with the predicted digit.

        Give it a try and see how accurately it can recognize handwritten digits!


        """
    )
    st.header("**𝗨𝗽𝗹𝗼𝗮𝗱 𝗮𝗻 𝗶𝗺𝗮𝗴𝗲 𝗼𝗳 𝗮 𝗵𝗮𝗻𝗱𝘄𝗿𝗶𝘁𝘁𝗲𝗻 𝗱𝗶𝗴𝗶𝘁**")
    uploaded_file = st.file_uploader("𝗦𝗨𝗕𝗠𝗜𝗧 𝗜𝗠𝗔𝗚𝗘", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        max_size = (200, 200)
        st.image(image, caption='Uploaded Image', use_column_width=True,width=max_size)
        st.write("")
        st.write("Classifying...")

        digit = recognize_digit(image)
        if digit is not None:
            st.success(f"𝗧𝗵𝗶𝘀 𝗱𝗶𝗴𝗶𝘁 𝗶𝘀 𝗽𝗿𝗼𝗯𝗮𝗯𝗹𝘆 𝗮 {digit}")

if __name__ == "__main__":
    main()
