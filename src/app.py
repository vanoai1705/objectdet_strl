import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os

from const import CLASSES, COLORS
from settings import DEFAULT_CONFIDENCE_THRESHOLD, DEMO_IMAGE, MODEL, PROTOTXT


@st.cache
def process_image(image):
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    net.setInput(blob)
    detections = net.forward()
    return detections


@st.cache
def annotate_image(
    image, detections, confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD
):
    # loop over the detections
    (h, w) = image.shape[:2]
    labels = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # prediction
            label = f"{CLASSES[idx]}: {round(confidence * 100, 2)}%"
            labels.append(label)
            cv2.rectangle(image, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(
                image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2
            )
    return image, labels

st.title("NGUYEN VAN OAI - 19110421")
st.title("OBJECT DETECTION")

img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
confidence_threshold = st.slider(
    "Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05
)

if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))

else:
    demo_image = DEMO_IMAGE
    image = np.array(Image.open(demo_image))

detections = process_image(image)
image, labels = annotate_image(image, detections, confidence_threshold)

st.image(
    image, caption=f"Processed image", use_column_width=True,
)

st.write(labels)

st.sidebar.header("PROJECT MACHINE LEARNING")
st.sidebar.image('https://bkhost.vn/wp-content/uploads/2022/07/Machine-Learning.jpg')
st.sidebar.image('https://som.edu.vn/wp-content/uploads/2022/11/machine-learning-la-gi-cach-hoat-dong-phan-loai-ung-dung.jpg')
st.sidebar.image('https://www.freecodecamp.org/news/content/images/2022/08/A6125B75-DB79-4448-94C9-E6ABD3E0E3E9.jpeg')
st.sidebar.image('https://media.springernature.com/w580h326/nature-cms/uploads/collections/AI_and_machine_learning-00afb90f3d21234fd0f207243f60aa8e.jpg')

