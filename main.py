import streamlit as st
import numpy as np
from PIL import Image
import pickle
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA


llm = ChatGroq(
    temperature=0, 
    groq_api_key='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', 
    model_name="llama-3.1-8b-instant",
)

with open('CNN.pickle', 'rb') as f:
    model = pickle.load(f)

class_names = ['Healthy Coral','Bleached Coral']

def preprocess_image(image, target_size=(256, 256)):
    try:
        img = Image.open(image).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        raise ValueError(f"Error in preprocessing image: {e}")
    
    
def generate_message(predicted_class):
    messages = [
        (
            "system",
            f"You are a knowledgeable assistant who provides detailed, user-friendly explanations. "
            f"Explain what {predicted_class} is, including its characteristics, significance, and common use cases."
        ),
    ]
    ai_msg = llm.invoke(messages)
    res = ai_msg.content
    return res

def predict_image(model, image_array):
    try:
        # Make the prediction
        predictions = model.predict(image_array)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = round(100 * np.max(predictions[0]), 2)
        return predicted_class

    except Exception as e:
        raise ValueError(f"An error occurred: {str(e)}")


def add_logo(logo_path, width=150):
    st.markdown(
        f"""
        <div style="display: flex; margin-bottom: 20px;">
            <img src="data:image/png;base64,{convert_image_to_base64(logo_path)}" alt="Logo" style="width: {width}px;">
        </div>
        """,
        unsafe_allow_html=True,
    )

def convert_image_to_base64(image_path):
    import base64
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def create_streamlit_app():
    st.markdown(
        """
        <style>
        .main {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
        
        h2 {
            margin-top: 7px;
        }
        
        h4 {
            margin-top: 13px;
        }
        
        h3, p {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    add_logo("logo.png", width=340)
    
    st.markdown("<h3 style='font-size: 20px; margin-bottom: -30px;'>Upload a coral image to evaluate its health and contribute to marine conservation.</h3>", unsafe_allow_html=True)
    uploaded_image = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", width=400)

    if st.button("Classify"):
        if uploaded_image is not None:
            try:
                image_array = preprocess_image(uploaded_image)
                predicted_class = predict_image(model, image_array)
                explanation_message = generate_message(predicted_class)
                explanation_message = explanation_message.replace("**", "<strong>").replace("</strong><strong>", "</strong>")
                st.markdown(f"<h3 style='font-size: 28px; font-weight: bold; '>Result: {predicted_class}</h3>", unsafe_allow_html=True)
                st.markdown(f"<h4 style='font-size: 24px; line-height: 1.6;'> {explanation_message}</h4>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Please upload an image before classification.")

    st.markdown("<h6 style='font-size: 28px; margin-top: 90px; text-align: center;'>About Us</h6>", unsafe_allow_html=True)
    
    st.markdown("""
    <p style='font-size: 18px; line-height: 1.2; text-align: center;'>
        Join us in safeguarding marine life with <strong>CoralGuard</strong>, an innovative app that helps preserve the 
        beauty and vitality of coral ecosystems. By leveraging cutting-edge AI <br/> technology, CoralGuard enables 
        accurate coral health assessment, empowering users to make meaningful contributions to global marine 
        conservation efforts. Together, let’s protect our underwater world and ensure its 
        future for generations to come.
        <br/>
        <br/>
    <strong>CoralGuard</strong> is an innovative project developed by <strong>Kshitij Angurala</strong> and <strong>Kanishka</strong> at 
    UPES Dehradun, demonstrating their exploration of the latest advancements in deep learning and computer vision. The 
    development process included extensive research and comparisons of various models, such as Convolutional Neural Networks
    (CNNs), Attention Mechanisms, and region-based approaches like Fast R-CNN. This approach underscores the team's commitment 
    to utilizing cutting-edge technology for marine conservation.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style='font-size: 13px; text-align: center; margin-top: 2px; bottom: 0; width: 100%; padding: 10px;'>
        © 2024, <strong>CoralGuard</strong>, Inc. or its affiliates. All rights reserved.
    </p>
    """, unsafe_allow_html=True)



if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="CoralGuard", page_icon="logo1.png")
    create_streamlit_app()
