import numpy as np
import requests
import light_side as ls
import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison


def main():
    # pylint: disable=no-member

    st.title("Light Side of the Night Demo Page")
    st.write(
        "**Light Side** is an low-light image enhancement library  that consist state-of-the-art deep learning methods. The light side of the Force is referenced. The aim is to create a light structure that will find the `Light Side of the Night`."
    )

    url = "https://raw.githubusercontent.com/canturan10/light_side/master/src/light_side.png?raw=true"
    light_side = Image.open(requests.get(url, stream=True).raw)
    st.sidebar.image(light_side, width=150)
    st.sidebar.title("light_side")
    st.sidebar.caption(f"Version `{ls.__version__}`")

    uploaded_file = st.sidebar.file_uploader(
        "", type=["png", "jpg", "jpeg"], accept_multiple_files=False
    )

    selected_model = st.sidebar.selectbox(
        "Select model",
        ls.available_models(),
    )
    selected_version = st.sidebar.selectbox(
        "Select version",
        ls.get_model_versions(selected_model),
    )

    if uploaded_file is None:
        # Default image.
        url = "https://raw.githubusercontent.com/canturan10/lsellighte/master/src/eurols_samples/HerbaceousVegetation.jpg?raw=true"
        image = Image.open(requests.get(url, stream=True).raw)

    else:
        # User-selected image.
        image = Image.open(uploaded_file)

    st.write("### Inferenced Image")
    image = np.array(image.convert("RGB"))

    model = ls.Enhancer.from_pretrained(selected_model, selected_version)
    model.eval()

    results = model.predict(image)[0]
    orj_img = results["image"]
    enh_img = results["enhanced"]

    image_comparison(
        img1=orj_img,
        img2=enh_img,
    )


if __name__ == "__main__":
    main()
