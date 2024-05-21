import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageFilter
import SimpleITK as sitk

import pandas as pd
import numpy as np
import pickle
import keras
import cv2
from scipy.ndimage import binary_fill_holes

from functions import featurexImg, load_image


# function to enable file uploader when the uploaded file is deleted
def enable():
    st.session_state.disabled=False


# To hide the sidebar
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)

# Reduce padding
st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 1rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

st.title('CT Analysis')
st.markdown(f"""<p style="font-size: 20px;">Welcome <b>{st.session_state["name"]}</b>!</p>""",
            unsafe_allow_html=True)

if st.button("Logout"):
    st.session_state.authentication_status = None
    st.switch_page("login.py")

# Specify canvas parameters in application
col1, col2 = st.columns([0.25,0.75], gap='large')
with col1:
    # To avoid double upload
    if 'disabled' not in st.session_state:
        st.session_state.disabled = False

    bg_image = st.file_uploader("Choose image to segment:", type=["png", "jpg"], disabled=st.session_state.disabled, on_change=enable)

    if bg_image is not None:
        st.session_state.disabled = True
        auto_generated = False  # variable to control whether the mask is automatically generated or not

        size_up = 512
        img = Image.open(bg_image).resize((size_up,size_up))
        # img2 = cv2.resize(np.array(img), (512, 512))
        width, height = img.size
        drawing_mode = st.selectbox(
            "Drawing tool:", ("freedraw", "polygon") #  "line", "rect", "circle", "point", "transform"
        )

        stroke_width = 3   # st.slider("Stroke width: ", 1, 5, 3)
        # if drawing_mode == 'point':
        #    point_display_radius = st.slider("Point display radius: ", 1, 5, 3)
        stroke_color = st.color_picker("Stroke color hex: ", "#E5FB11")

        realtime_update = True  # st.checkbox("Update in realtime", True)

        with col2:
            # Push the content of the right column upper
            st.markdown("""
                    <style>
                           .st-emotion-cache-165v49w {
                                position: relative;
                                bottom: 4rem;
                            }
                    </style>
                    """, unsafe_allow_html=True)


            st.markdown('<p style="font-size: 20px;">Click here to automatically generate the mask.</p>',
                        unsafe_allow_html=True)

            # Automatically segmented mask
            if st.button("Generate mask"):
                # Load the segmentation model
                #with open('C:/Users/momiv/OneDrive/Desktop/magistrale/1st_year/SIAM in Healthcare/project2/pages/segmUnet.pkl','rb') as f:
                    # unet = pickle.load(f)
                unet = keras.models.load_model('/Users/davidevettore/PycharmProjects/Healthcare/pages/segmUnet100_aug.keras')
                size_down = 128
                threshold = 1e-1
                #threshold = 0.5
                segm = load_image(np.array(img),size_down)
                # st.image(segm)

                # pred = unet.predict(np.expand_dims(segm, 0), verbose=0)[0] > threshold
                mask = unet.predict(np.expand_dims(segm, 0), verbose=0)[0] > threshold

                # print(pred.shape)
                auto_generated = True
                # print(pred.sum())

            st.markdown('<p style="font-size: 20px;">Segment the image to obtain a mask.</p>',
                        unsafe_allow_html=True)
            # st.write("Segment the image to obtain a mask.")
            if drawing_mode == "polygon":
                st.write("Click to add vertices, right click to close the polygon.")
            # Create a canvas component
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
                stroke_width=stroke_width,
                stroke_color=stroke_color,
                background_image=img if bg_image else None,
                update_streamlit=realtime_update,
                width=width,
                height=height,
                drawing_mode=drawing_mode,
                #point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
                key="canvas",
            )
            st.markdown('<p style="font-size: 20px;">The mask will appear here:</p>',
                        unsafe_allow_html=True)
            # Do something interesting with the image data and paths
            if canvas_result.image_data is not None or auto_generated:
                # st.image(canvas_result.image_data)
                if canvas_result.image_data is not None and not auto_generated:
                    mask = canvas_result.image_data
                    mask = mask[:, :, -1] > 0
                # print(np.count_nonzero(mask))
                    if drawing_mode == "freedraw":
                        mask = binary_fill_holes(mask)

                # st.image(mask)
                # print(mask.sum())
                if mask.sum() > 0:

                    if auto_generated:
                        mask_todisp = Image.fromarray(mask[:, :, -1]).resize((size_up, size_up), resample=Image.Resampling.LANCZOS)
                        mask_todisp = mask_todisp.convert("L")
                        # Blurring to smooth the edges
                        mask_todisp = mask_todisp.filter(ImageFilter.GaussianBlur(radius=5))

                        # Thresholding
                        threshold_value = 175  # You can adjust this value based on your needs
                        mask_todisp = mask_todisp.point(lambda p: 255 if p > threshold_value else 0)

                        mask = np.array(mask_todisp)/255
                        print("auto")
                    else:
                        mask_todisp = Image.fromarray(mask)
                        print("Not auto")

                    print(mask.shape)
                    st.image(mask_todisp)

                    mask_sitk = sitk.GetImageFromArray(mask.astype(np.uint8))
                    img_sitk = sitk.GetImageFromArray(np.array(img).astype(np.float32))

                    feat = featurexImg(img_sitk[0,:,:], mask_sitk, ["firstorder","shape2D","glcm", "gldm", "glrlm", "glszm", "ngtdm"] ) # ["firstorder","shape2D","glcm", "gldm", "glrlm", "glszm", "ngtdm"]
                    feat = feat.drop(["original_glcm_SumAverage"], axis=1)
                    # print(type(feat))
                    # print(feat.shape)
                    # print(feat)

                    # Normalization
                    norm_df = pd.read_csv("/Users/davidevettore/PycharmProjects/Healthcare/normalization_all.csv")
                    mean = norm_df["mean"]
                    std = norm_df["std"]
                    mean.index = feat.columns.tolist()
                    std.index = feat.columns.tolist()
                    feat.loc["mean"] = mean
                    feat.loc["std"] = std
                    # print(mean.shape)
                    # print(std.shape)
                    norm_feat = (feat.iloc[0] - feat.loc['mean']) / feat.loc['std']
                    # print(norm_feat)

                    # Load the model
                    with open('/Users/davidevettore/PycharmProjects/Healthcare/pages/svm_all.pkl', 'rb') as f:
                        svm_classifier = pickle.load(f)

                    # Make prediction
                    # pred = svm_classifier.predict(np.array(norm_feat).reshape(1, -1))[0] # class prediction
                    # Vector of probabilities
                    probs = svm_classifier.predict_proba(np.array(norm_feat).reshape(1, -1))[0]
                    pred = np.argmax(probs)

                    if pred == 0:
                        print(f"The selected region is benign with a probability of {probs[0]*100:.2f}%.")
                    elif pred == 1:
                        print(f"The selected region is malignant with a probability of {probs[1]*100:.2f}%.")





                    # print(type(img)) # <class 'PIL.PngImagePlugin.PngImageFile'>
                    # print(type(mask_todisp)) # <class 'PIL.Image.Image'>
                    # print(type(mask_sitk)) # <class 'SimpleITK.SimpleITK.Image'>
    # if canvas_result.json_data is not None:
    #     objects = pd.json_normalize(canvas_result.json_data["objects"])  # need to convert obj to str because PyArrow
    #     for col in objects.select_dtypes(include=['object']).columns:
    #         objects[col] = objects[col].astype("str")
    #     st.dataframe(objects)

# print(canvas_result.image_data.shape)   # (height, width, 4)
# print(canvas_result.json_data)


# TODO: compute DICE

# TODO: 1 region per mask



