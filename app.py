# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # import tensorflow as tf
# # import os
# # from dotenv import load_dotenv
# # import subprocess
# # import subprocess
# # load_dotenv()
# # print("Installed Packages:")
# # subprocess.run(["pip", "list"])




# # print("Python Path:", os.sys.executable)
# # print("Installed Packages:")
# # subprocess.run(["pip", "list"])
# # port = int(os.getenv('PORT',8501))
# # model = tf.keras.models.load_model('credit_card_fraud_detection_model.h5')
# # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # st.title("Credit Card Fraud Detection")
# # st.write("This app predicts whether a transaction is fraudulent based on uploaded CSV data.")


# # st.header("Upload CSV File")
# # uploaded_file = st.file_uploader("Upload a CSV file containing transaction data", type=["csv"])

# # def predict_fraud(input_data):
# #     predictions = model.predict(input_data)
# #     return ["Fraud" if pred[0] > 0.5 else "Not Fraud" for pred in predictions]

# # if uploaded_file:
# #     try:
        
# #         st.write("Reading uploading file...")
# #         df = pd.read_csv(uploaded_file)
# #         st.write("File read successfully")
        
# #         required_columns = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
# #         if not all(col in df.columns for col in required_columns):
# #             st.error("CSV file must contain columns: Time, Amount, and V1 to V28.")
# #         else:
# #             st.write("Processing input data...")
# #             input_data = df[required_columns].values
            
# #             st.write("Making predictions...")
# #             df["Prediction"] = predict_fraud(input_data)
            
          
# #             st.subheader("Prediction Results")
# #             st.dataframe(df[["Time", "Amount", "Prediction"]])
            
      
# #             csv = df.to_csv(index=False).encode('utf-8')
# #             st.download_button(
# #                 label="Download Predictions as CSV",
# #                 data=csv,
# #                 file_name="fraud_predictions.csv",
# #                 mime="text/csv"
# #             )
# #     except Exception as e:
# #         st.error(f"Error processing file: {e}")
# #         st.title("My streamlit App")
# #         st.write(f"Running on PORT {port}")


# import streamlit as st
# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Load the model only once
# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model('credit_card_fraud_detection_model.h5')

# model = load_model()
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# st.title("Credit Card Fraud Detection")
# st.write("This app predicts whether a transaction is fraudulent based on uploaded CSV data.")

# st.header("Upload CSV File")
# uploaded_file = st.file_uploader("Upload a CSV file containing transaction data", type=["csv"])

# def predict_fraud(input_data):
#     predictions = model.predict(input_data)
#     return ["Fraud" if pred[0] > 0.5 else "Not Fraud" for pred in predictions]

# if uploaded_file:
#     try:
#         st.write("Reading uploaded file...")
#         df = pd.read_csv(uploaded_file)
#         st.write("File read successfully.")
        
#         required_columns = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
#         if not all(col in df.columns for col in required_columns):
#             st.error("CSV file must contain columns: Time, Amount, and V1 to V28.")
#         else:
#             st.write("Processing input data...")
#             input_data = df[required_columns].values
            
#             st.write("Making predictions...")
#             df["Prediction"] = predict_fraud(input_data)
            
#             st.subheader("Prediction Results")
#             st.dataframe(df[["Time", "Amount", "Prediction"]])
            
#             csv = df.to_csv(index=False).encode('utf-8')
#             st.download_button(
#                 label="Download Predictions as CSV",
#                 data=csv,
#                 file_name="fraud_predictions.csv",
#                 mime="text/csv"
#             )
#     except Exception as e:
#         st.error(f"Error processing file: {e}")

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from dotenv import load_dotenv

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load environment variables
load_dotenv()

# Load the model only once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('credit_card_fraud_detection_model.h5')

model = load_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

st.title("Credit Card Fraud Detection")
st.write("This app predicts whether a transaction is fraudulent based on uploaded CSV data.")

st.header("Upload CSV File")
uploaded_file = st.file_uploader("Upload a CSV file containing transaction data", type=["csv"])

def predict_fraud(input_data):
    predictions = model.predict(input_data)
    return ["Fraud" if pred[0] > 0.5 else "Not Fraud" for pred in predictions]

if uploaded_file:
    try:
        st.write("Reading uploaded file...")
        df = pd.read_csv(uploaded_file)
        st.write("File read successfully.")
        
        required_columns = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
        if not all(col in df.columns for col in required_columns):
            st.error("CSV file must contain columns: Time, Amount, and V1 to V28.")
        else:
            st.write("Processing input data...")
            input_data = df[required_columns].values
            
            st.write("Making predictions...")
            df["Prediction"] = predict_fraud(input_data)
            
            st.subheader("Prediction Results")
            st.dataframe(df[["Time", "Amount", "Prediction"]])
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )
    except Exception as e:
        st.error(f"Error processing file: {e}")