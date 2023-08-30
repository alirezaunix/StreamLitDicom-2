import streamlit as st
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Add, PReLU, Conv2DTranspose, Concatenate, MaxPooling2D, UpSampling2D, Dropout, \
    Activation, Subtract
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
import os
import pydicom
import cv2
import numpy as np
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, Subtract
from tensorflow.keras.layers import Input
import glob
from PIL import Image
import os

currentDir=os.path.abspath(os.path.dirname(__name__))

# Function to get DNCNN model
def get_dncnn_model(input_channel_num):
    inpt = Input(shape=(None, None, input_channel_num))
    # 1st layer, Conv+relu
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(inpt)
    x = Activation('relu')(x)
    # 15 layers, Conv+BN+relu
    for i in range(15):
        x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)
    # last layer, Conv
    x = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
    x = Subtract()([inpt, x])  # input - noise
    model = Model(inputs=inpt, outputs=x)
    return model

# Function to get the desired model
def get_model(input_channel_num, model_name="srresnet"):
    if model_name == "srresnet":
        return get_srresnet_model(input_channel_num=input_channel_num)
    elif model_name == "unet":
        return get_unet_model(input_channel_num=input_channel_num, out_ch=input_channel_num)
    elif model_name == "dncnn":
        return get_dncnn_model(input_channel_num=input_channel_num)
    else:
        raise ValueError("model_name should be 'srresnet'or 'unet'")

# Function to load and preprocess DICOM image
def load_dicom_image(file):
    ds = pydicom.dcmread(file)
    img = (ds.pixel_array)[160:890, 160:890]
    img = np.clip(img / factor, 0, 255)
    img = np.uint8(img)
    return img

def save_image(img, filename):
    cv2.imwrite(filename, img)

factor = 7

# Streamlit web app UI
def main():
    # Load the logo image
    global currentDir
    logo_image = Image.open(currentDir+'/download.jpg')

    # Display the logo image
    st.image(logo_image, caption='QMISG')
    # st.title("Image Denoising App")
    st.markdown("<h1 style='text-align: center; color: #00b8e6;'>Image Denoising App</h1>", unsafe_allow_html=True)
    st.markdown(
        "<h2 style='text-align: center; color: #565656;'>Choose a DICOM image for denoising</h2>",
        unsafe_allow_html=True)

    # Select DICOM image using file uploader
    uploaded_file = st.file_uploader("Upload DICOM Image")

    if uploaded_file is not None:
        # Load and preprocess uploaded image
        img = load_dicom_image(uploaded_file)

        # Load and preprocess model
        weight_file = "center_weights.011-50.997-29.16076_dncnn_CD_fantom.hdf5"
        input_channel_num = 1
        model = get_model(input_channel_num=input_channel_num, model_name="dncnn")
        model.load_weights(weight_file)

        # Denoise image
        pred = model.predict(np.expand_dims(np.expand_dims(img, 2), 0))
        denoised_image = np.clip(pred[0][:, :, 0], 0, 255).astype(dtype=np.uint8)

        # Display original and denoised images
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(img, use_column_width=True)
        with col2:
            st.subheader("Denoised Image")
            st.image(denoised_image, use_column_width=True)

        # Save image button
        if st.button("Save Denoised Image"):
            address = st.text_input("Enter the address to save the image:")
            save_image(denoised_image, address + "/denoised_image.png")
            st.success("Denoised image saved successfully!")

if __name__ == "__main__":
    main()

# Add footer
st.write('Developed by QMISG')