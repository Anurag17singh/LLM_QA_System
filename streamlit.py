import streamlit as st

st.title("Hotel Booking QA System")

query = st.text_input("Enter your question:")

if st.button("Submit"):
    response = requests.post("http://127.0.0.1:5000/query", json={"query": query}).json()
    st.write("Answer:", response["answer"])