import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.title("Food Detector")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()
conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)

source = st.radio("Input", ["Upload image", "Camera"])
file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"]) if source == "Upload image" else st.camera_input("Take a photo")

if file is not None:
    img = Image.open(file).convert("RGB")
    r = model(img, conf=conf, verbose=False)[0]
    st.image(r.plot()[..., ::-1], caption="Detections", use_container_width=True)
    if len(r.boxes) == 0:
        st.info("No food detected at this threshold.")
    for b in r.boxes:
        st.write(f"{r.names[int(b.cls[0])]}: {float(b.conf[0]):.2f}")