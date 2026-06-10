import streamlit as st
import pickle

# ---Load Model------
model=pickle.load(open('spam_model.pkl','rb'))
vectorizer=pickle.load(open('vectorizer.pkl','rb'))

# -----Page config-----------
st.set_page_config(page_title="Spam Detection App",layout="centered")

# -----Title---------
st.title("📧 Email Spam Detection")
st.write("Enter an email message below to check if it spam or not.")

# ------User Input-------
message=st.text_area("Enter message")

# -----Button-----------
if st.button("Predict"):
    if message.strip()=="":
        st.warning("Please enter a message to predict")
    else:
        transformed_message=vectorizer.transform([message])

        # --prediction----
        prediction=model.predict(transformed_message)[0]


        st.subheader("Prediction Result")

        if prediction=="spam":
            st.error("🚨This message is classified as SAPM")
        else:
            st.success("✅ This message is classified as Not SPAM")