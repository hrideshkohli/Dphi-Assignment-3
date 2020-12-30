import pycaret
from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np

model=load_model('saved_model_assignment3')

def predict(model, input_df):
    predictions_df=predict_model(model, data=input_df)
    predictions=predictions_df['Label'][0]
    
    return predictions


def run():
    from PIL import Image
    image=Image.open('image.jpg')
    
    
    st.sidebar.info('This app is created to predict Loan approval')
    
    st.sidebar.image(image)
    
    st.title("Loan Prediction App by Hridesh Kohli")
    
    
    gender = st.selectbox('Sex', ['Male', 'Female'])
    married=st.selectbox('Married', ['Yes', 'No'])
    dependents=st.selectbox('Dependents', ['0', '1', '2', '3+'])
    education=st.selectbox('Education', ['Graduate', 'Not Graduate'])
    selfemployed=st.selectbox('Self Employed', ['Yes', 'No'])
    applicantincome=st.number_input('Applicant Income', min_value=100, max_value=35000, value=500)
    coapplicantincome=st.number_input('Co-Applicant Income', min_value=0, max_value=20000, value=500)
    loanamount=st.number_input('Loan Amount', min_value=0.0, max_value=600.0, value=200.0)
    loanamount_term=st.number_input('Loan Amount Term', min_value=60.0, max_value=500.0, value=360.0)
    credithistory=st.selectbox('Credit History', ['0.0', '1.0'])
    propertyarea=st.selectbox('Property Area', ['Semiurban', 'Urban','Rural'])
    
    input_dict1={'Loan_ID': "LP001116", 'Gender': "Male", 'Married': "No", 'Dependents': 0,
    'Education':"Not Graduate", 'Self_Employed': "No", 'ApplicantIncome':3748,
    'CoapplicantIncome': 1668.0,'LoanAmount': 110.0, 'Loan_Amount_Term': 360.0,
    'Credit_History':1.0, 'Property_Area': "Semiurban" }
    
    input_dict={'Loan_ID': "LP001116", 'Gender': gender, 'Married': married, 'Dependents': dependents,
    'Education':education, 'Self_Employed': selfemployed, 'ApplicantIncome':applicantincome,
    'CoapplicantIncome': coapplicantincome,'LoanAmount': loanamount, 'Loan_Amount_Term': loanamount_term,
    'Credit_History':1.0, 'Property_Area': propertyarea}
    
    input_df=pd.DataFrame([input_dict])
    
    output=predict(model, input_df)
    
    if st.button("Predict"):
        output=predict(model=model, input_df=input_df)
        
        #output = str(output)
        #st.success(input_df.columns)
        #st.success()
        if output==1:
            results="High probability that LOAN will be approved"
        else:
            results="Sorry! You have low changes of getting LOAN approval"
        
        st.success(str(output))
        st.success(results)
    
    
if __name__ == '__main__':
    run()    
    
    