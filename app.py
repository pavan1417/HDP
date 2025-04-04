import streamlit as st

# Title
st.title("Heart Health Form")

# Create form fields
with st.form(key="heart_health_form"):
    age = st.number_input("Age", min_value=0, step=1)
    gender = st.selectbox("Gender", ["Select Gender", "Female", "Male", "Other"])
    chest_pain = st.selectbox("Chest Pain Level", [0, 1, 2, 3, 4])
    blood_pressure = st.number_input("Blood Pressure", min_value=0, step=1)
    years_smoking = st.number_input("Years of Smoking", min_value=0, step=1)
    fasting_sugar = st.number_input("Fasting Blood Sugar", min_value=0, step=1)
    diabetes_history = st.selectbox("Diabetes History", ["Select", "Yes", "No"])
    family_history = st.selectbox("Family History of Heart Disease", ["Select", "Yes", "No"])
    ecg = st.number_input("ECG", min_value=0, step=1)
    pulse_rate = st.number_input("Pulse Rate", min_value=0, step=1)
    
    # Submit button
    submit_button = st.form_submit_button("Submit")

if submit_button:
    st.subheader("Submitted Details:")
    st.write(f"**Age:** {age}")
    st.write(f"**Gender:** {gender}")
    st.write(f"**Chest Pain Level:** {chest_pain}")
    st.write(f"**Blood Pressure:** {blood_pressure}")
    st.write(f"**Years of Smoking:** {years_smoking}")
    st.write(f"**Fasting Blood Sugar:** {fasting_sugar}")
    st.write(f"**Diabetes History:** {diabetes_history}")
    st.write(f"**Family History of Heart Disease:** {family_history}")
    st.write(f"**ECG:** {ecg}")
    st.write(f"**Pulse Rate:** {pulse_rate}")
