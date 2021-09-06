from PIL import Image
import streamlit as st
# import numpy as np
# import pandas as pd

st.set_page_config(
    page_title="Waste Classifier",
    page_icon="♻️",
    layout="centered",  # wide
    initial_sidebar_state="auto")  # collapsed

st.markdown("""# ♻️ WASTE CLASSIFIER ♻️


#### “There is no such thing as ‘away’. When we throw anything away it must go somewhere.”
– Annie Leonard""")

st.text("")
st.text("")
"""Welcome to our waste classifier. We can tell you if your waste is..."""

columns = st.columns(6)
cardboard_image = Image.open('images/cardboard44.jpg')
cardboard = columns[0].image(cardboard_image, caption='Cardboard', use_column_width=True)

glass_image = Image.open('images/glass9.jpg')
glass= columns[1].image(glass_image, caption='Glass', use_column_width=True)

metal_image = Image.open('images/metal28.jpg')
metal = columns[2].image(metal_image, caption='Metal', use_column_width=True)

paper_image = Image.open('images/paper24.jpg')
paper = columns[3].image(paper_image, caption='Paper', use_column_width=True)

plastic_image = Image.open('images/plastic14.jpg')
plastic = columns[4].image(plastic_image, caption='Plastic', use_column_width=True)

trash_image = Image.open('images/trash113.jpg')
trash = columns[5].image(trash_image, caption='Trash', use_column_width=True)
st.text("")
st.text("")
st.set_option('deprecation.showfileUploaderEncoding', False)

uploaded_file = st.file_uploader("Choose a image of waste for classification:", type="JPG")

if uploaded_file is not None:
    import time
    'Activating neural networks...'

    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(20):
        latest_iteration.text(f'Iteration {i+1}')
        bar.progress(i + 1)
        time.sleep(0.1)

    st.success('Image classified!')
st.text("")
st.text("")
st.text("")

st.markdown("# Our Data")
"""Many thanks to Pedro Proença and Pedro Simões of tacodataset.org as well as Gary Thung of trashnet for thier invaluable open source data."""

#TEAM
st.markdown("# Our Team")
columns = st.columns(4)
xin_image = Image.open("images/xin.jpeg")
xin = columns[2].image(xin_image, caption='Xin Zhan', use_column_width=True)
lucy_image = Image.open("images/lucy.jpeg")
lucy = columns[0].image(lucy_image, caption='Lucy Meyler', use_column_width=True)
jack_image = Image.open("images/jack.jpeg")
jack = columns[1].image(jack_image, caption='Jack Riddleston', use_column_width=True)
izzy_image = Image.open("images/izzy.JPG")
izzy = columns[3].image(izzy_image, caption='Izzy Weber', use_column_width=True)

st.sidebar.write(f"""
    # Navigation
    [The Classifier](//#waste-classifier)\n
    [Our Team](/#our-team)\n
    [Our Data](/#our-data)\n
    """)

