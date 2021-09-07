from PIL import Image
import streamlit as st
import numpy as np
import pandas as pd
import time

st.set_page_config(
    page_title="Waste Classifier",
    page_icon="♻️",
    layout="centered",  # wide
    initial_sidebar_state="auto")  # collapsed

st.write('<a name=" ♻️ Waste Classifier ♻️"></a>', unsafe_allow_html=True)
st.markdown("# ♻️ Waste Classifier ♻️")

"""#### “There is no such thing as ‘away’. When we throw anything away it must go somewhere.”
– Annie Leonard"""

st.text("")
st.text("")
"""Welcome to our waste classifier. We can tell you if your waste is..."""

columns = st.columns(6)
new_size = (360, 360)
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
    beep = "Beep"
    boop = "Boop"
    X_test = Image.open(uploaded_file)
    new_size = (180, 180)
    X_test= X_test.resize(new_size)
    columns = st.columns(2)
    time.sleep(1)
    columns[0].write("Resizing image... new size is:")
    columns[0].write(X_test.size)
    npframe = np.array(X_test.getdata())
    imgrgb_df = pd.DataFrame(npframe)
    columns[0].image(X_test)
    imgrgb_df = imgrgb_df.to_numpy().reshape((180, 180, 3))
    time.sleep(1)
    columns[1].write("Reshaping image... new shape is:")
    columns[1].write(imgrgb_df.shape)
    time.sleep(1)
    columns[1].write('Activating neural networks...')
    time.sleep(1)
    columns[1].write(beep)
    time.sleep(1)
    columns[1].write(beep)
    time.sleep(1)
    columns[1].write(boop)
    #model = load_model()
    #prediction = model.predict(imgrgb_df)
    st.markdown(f'# Your waste is a !')

st.text("")

st.write('<a name=" Our Data"></a>', unsafe_allow_html=True)
st.markdown("# Our Data")
"""Many thanks to Pedro Proença and Pedro Simões of tacodataset.org as well as Gary Thung of trashnet for their invaluable open source data."""

st.write('<a name=" Our Team"></a>', unsafe_allow_html=True)
st.markdown("# Our Team")
columns = st.columns(4)
xin_image = Image.open("images/xin.jpeg")
xin = columns[0].image(xin_image, caption='Xin Zhan', use_column_width=True)
lucy_image = Image.open("images/lucy.jpeg")
lucy = columns[1].image(lucy_image, caption='Lucy Meyler', use_column_width=True)
jack_image = Image.open("images/jack.jpeg")
jack = columns[2].image(jack_image, caption='Jack Riddleston', use_column_width=True)
izzy_image = Image.open("images/izzy.JPG")
izzy = columns[3].image(izzy_image, caption='Izzy Weber', use_column_width=True)

left_nav_toc = '''<h1>Navigation</h1>
            <a href="# ♻️ Waste Classifier ♻️">The Classifier</a><br>
                        <a href="# Our Data">Our Data</a><br>
            <a href="# Our Team">Our Team</a><br>
        '''

st.sidebar.markdown(left_nav_toc, unsafe_allow_html=True)
