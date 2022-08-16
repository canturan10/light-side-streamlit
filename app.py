import random
from datetime import datetime

import av
import light_side as ls
import numpy as np
import requests
import streamlit as st
from PIL import Image
from streamlit_image_comparison import image_comparison
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer


def main():
    # pylint: disable=no-member

    st.set_page_config(
        page_title="Light Side Demo Page",
        page_icon="⚡️",
        layout="centered",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://canturan10.github.io/light_side/",
            "About": "Low-Light Image Enhancement",
        },
    )
    st.title("Light Side Demo Page")

    url = "https://raw.githubusercontent.com/canturan10/light_side/master/src/light_side.png?raw=true"
    light_side = Image.open(requests.get(url, stream=True).raw)

    st.sidebar.image(light_side, width=100)
    st.sidebar.title("Light Side of the Night")
    st.sidebar.caption(ls.__description__)

    st.sidebar.write(
        "**Light Side** is an low-light image enhancement library  that consist state-of-the-art deep learning methods. The light side of the Force is referenced. The aim is to create a light structure that will find the `Light Side of the Night`."
    )

    st.sidebar.caption(f"Version: `{ls.__version__}`")
    st.sidebar.caption(f"License: `{ls.__license__}`")
    st.sidebar.caption("")
    st.sidebar.caption(f"[Website](https://lnkd.in/dKejmaH2)")
    st.sidebar.caption(f"[Docs](https://lnkd.in/dt-2T4ZN)")
    st.sidebar.caption(f"[Github](https://lnkd.in/dDttpiEG)")
    # st.sidebar.caption(f"[Demo Page](https://lnkd.in/dxy-632M)")
    st.sidebar.caption(f"[Hugging Face]( https://lnkd.in/d5-VVK62)")
    st.sidebar.caption(f"[Pypi](https://lnkd.in/d7eDNb-B)")
    st.sidebar.caption("")
    st.sidebar.caption(ls.__copyright__)

    selected_model = st.selectbox(
        "Select model",
        ls.available_models(),
    )
    selected_version = st.selectbox(
        "Select version",
        ls.get_model_versions(selected_model),
    )

    mode = st.radio("Select Inference Mode", ("Image", "Video (WebRTC)"))

    model = ls.Enhancer.from_pretrained(selected_model, selected_version)
    model.eval()

    if mode == "Image":
        uploaded_file = st.file_uploader(
            "", type=["png", "jpg", "jpeg"], accept_multiple_files=False
        )
        if uploaded_file is None:
            st.write("Default Image")
            # Default image.
            url = f"https://github.com/canturan10/light_side/blob/master/src/sample/{random_sample}?raw=true"
            image = Image.open(requests.get(url, stream=True).raw)

        else:
            # User-selected image.
            image = Image.open(uploaded_file)

        image = np.array(image.convert("RGB"))

        results = model.predict(image)[0]
        orj_img = results["image"]
        enh_img = results["enhanced"]

        image_comparison(
            img1=orj_img,
            img2=enh_img,
            label1="Dark Side",
            label2="Light Side",
        )
    else:

        st.write(
            "If video is not playing, please refresh the page. Depends on your browser and connection, it may take some time to load the video."
        )

        class VideoProcessor(VideoProcessorBase):
            def recv(self, frame):

                img = frame.to_ndarray(format="bgr24")
                results = model.predict(img)[0]
                orj_img = results["image"]
                enh_img = results["enhanced"]

                return av.VideoFrame.from_ndarray(
                    np.concatenate((orj_img, enh_img), axis=1), format="bgr24"
                )

        ctx = webrtc_streamer(
            key="example",
            video_processor_factory=VideoProcessor,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={
                "video": True,
                "audio": False,
            },
        )


if __name__ == "__main__":

    samples = [
        "0_orj.png",
        "1_orj.png",
        "2_orj.png",
        "3_orj.png",
        "4_orj.png",
        "5_orj.png",
    ]

    random.seed(datetime.now())
    random_sample = samples[random.randint(0, len(samples) - 1)]

    main()
