import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

model = pickle.load(open('GDP.pkl','rb'))
st.title('Graduation Admission Prediction')
def main():
    cgpa = st.slider("Input Your CGPA", 0.0, 10.0)
    gre = st.slider("Input your GRE Score", 0, 340)
    toefl = st.slider("Input your TOEFL Score", 0, 120)
    research = st.slider("Do You have Research Experience (0 = NO, 1 = YES)", 0, 1)
    uni_rating = st.slider("Rating of the University you wish to get in on a Scale 1-5", 1, 5)
    sop = st.slider("Rating of the SOP you get in on a Scale 1-5", 1.0, 5.0)
    lor = st.slider("Rating of the lor get in on a Scale 1-5", 1.0, 5.0)

    inputs = [[cgpa, gre, toefl, research, uni_rating, sop, lor]]

    if st.button('predict'):
        standardise_inputs = scaler.transform(inputs)
        result=model.predict(standardise_inputs)
        if result ==1:
            st.header('The candidate has high chance of getting admission.')
        else:
            st.header('Soory.The candidate has no chance of getting admission. ')

if __name__ == "__main__":
    main()



