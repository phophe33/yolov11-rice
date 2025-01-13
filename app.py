#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from PIL import Image
import streamlit as st

import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam
import pandas as pd
import numpy as np
import streamlit as st
# setting page layout
st.set_page_config(
    page_title="基于YOLOV11的水稻病虫害识别系统",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# main page heading
st.title("基于YOLOV11的水稻病虫害识别系统")

# sidebar
st.sidebar.header("DL Model Config")

# model options
task_type = st.sidebar.selectbox(
    "Select Task",
    ["Detection"]
)

model_type = None
if task_type == "Detection":
    model_type = st.sidebar.selectbox(
        "Select Model",
        config.DETECTION_MODEL_LIST
    )
else:
    st.error("Currently only 'Detection' function is implemented")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 0, 100, 50)) / 100

model_path = ""
if model_type:
    model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
else:
    st.error("Please Select Model in Sidebar")

# load pretrained DL model
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")

# image/video options
st.sidebar.header("Image/Video Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    config.SOURCES_LIST
)

source_img = None
if source_selectbox == config.SOURCES_LIST[0]: # Image
    infer_uploaded_image(confidence, model)
elif source_selectbox == config.SOURCES_LIST[1]: # Video
    infer_uploaded_video(confidence, model)
elif source_selectbox == config.SOURCES_LIST[2]: # Webcam
    infer_uploaded_webcam(confidence, model)
else:
    st.error("Currently only 'Image' and 'Video' source are implemented")

#######
if "graph" not in st.session_state:
    st.session_state.graph = ""

# 第一列
def column_1():
    st.header("1. 选择数据")
    st.selectbox(
        "选择数据?",
        (
            "bacterial blight",
            "blast",
            "brownspot",
            "tungro",
        ),
    )
    # 随机模拟的数据
    data = pd.DataFrame(np.random.randn(5, 3), columns=["x", "y", "confidence level"])
    st.table(data)


def column_2():
    st.header("2. 配置数据")
    graph = st.radio(
        "图形类型: ",
        ("折线图", "柱状图", "散点图"),
    )

    st.session_state.graph = graph


def column_3():
    st.header("3. 绘制图形")

    chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["x", "y", "confidence level"])
    if st.session_state.graph == "散点图":
        st.scatter_chart(chart_data)

    if st.session_state.graph == "折线图":
        st.line_chart(chart_data)

    if st.session_state.graph == "柱状图":
        st.bar_chart(chart_data)


col1, col2, col3 = st.columns(3)

with col1:
    column_1()

with col2:
    column_2()

with col3:
    column_3()
