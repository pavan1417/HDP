import streamlit as st
import requests

# Function to fetch data from the backend
def fetch_data():
    response = requests.get("http://localhost:5000/api/data")
    if response.status_code == 200:
        return response.json()
    return None

# Function to post data to the backend
def post_data(data):
    response = requests.post("http://localhost:5000/api/data", json=data)
    if response.status_code == 200:
        return response.json()
    return None

st.title("Streamlit Frontend with Flask Backend")

if st.button("Fetch Data"):
    data = fetch_data()
    if data:
        st.write(data["message"])
    else:
        st.error("Failed to fetch data from the backend")

input_data = st.text_input("Enter some data to send to the backend:")

if st.button("Send Data"):
    response = post_data({"data": input_data})
    if response:
        st.write(response)
    else:
        st.error("Failed to send data to the backend")
