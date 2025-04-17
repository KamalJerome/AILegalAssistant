import json
import streamlit as st
from langchain_together import Together
from PyPDF2 import PdfReader  # For PDF text extraction
from docx import Document as DocxDocument  # For Word documents
from PIL import Image  # For image handling
import pytesseract  # For OCR
import numpy as np
import keras
from keras.layers import GRU, Dense, Dropout, Input, Masking, Bidirectional
from keras.models import Model
import tensorflow as tf  # Import TensorFlow for math operations

# Set API key for LLM
api_key = '4787575f7f7a6e50baba23882aaea7a3a66486e2849b4c03d9e94fac6b25dfb1'
mistral_llm = Together(model="mistralai/Mixtral-8x22B-Instruct-v0.1", temperature=0.5, max_tokens=1024, together_api_key=api_key)

# Attention Layer for the Bi-GRU model
class AttentionLayer(keras.layers.Layer):
    def __init__(self, attention_dim=200, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim

    def build(self, input_shape):
        # Initialize weights for the attention mechanism
        self.W = self.add_weight(shape=(input_shape[-1], self.attention_dim), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(shape=(self.attention_dim,), initializer="zeros", trainable=True)
        self.u = self.add_weight(shape=(self.attention_dim, 1), initializer="glorot_uniform", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # Compute attention scores
        u_t = tf.math.tanh(tf.linalg.matmul(x, self.W) + self.b)  # Use tf.math.tanh and tf.linalg.matmul
        a = tf.linalg.matmul(u_t, self.u)
        a = tf.nn.softmax(tf.squeeze(a, -1))
        weighted_input = x * tf.expand_dims(a, -1)  # Ensure dimensions match
        return tf.reduce_sum(weighted_input, axis=1)

# Model loading and building function for Bi-GRU
def load_bi_gru_model():
    input_text = Input(shape=(None, 768), dtype='float32', name='text')
    masked_input = Masking(mask_value=-99.)(input_text)
    gru_out = Bidirectional(GRU(100, return_sequences=True))(masked_input)
    gru_out = Bidirectional(GRU(100, return_sequences=True))(gru_out)
    attention_out = AttentionLayer(attention_dim=200)(gru_out)
    dropout_out = Dropout(0.5)(attention_out)
    dense_out = Dense(30, activation='relu')(dropout_out)
    final_out = Dense(1, activation='sigmoid')(dense_out)
    model = Model(inputs=input_text, outputs=final_out)
    return model

# Initialize the Bi-GRU model
bi_gru_model = load_bi_gru_model()

# Streamlit interface
st.markdown("<h1 style='text-align: center;'>⚖️ AI Legal Assistant ⚖️</h1>", unsafe_allow_html=True)
st.write(" ")
st.subheader("👨‍⚖ Judgment Predictor", divider="grey")
st.write(" ")

# Document input function with OCR for scanned documents
def extract_text_from_file(file):
    text = ""
    if file.type == "application/pdf":
        pdf_reader = PdfReader(file)
        text = "\n\n".join([page.extract_text().strip() for page in pdf_reader.pages if page.extract_text()])  # Ensure proper spacing
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = DocxDocument(file)
        text = "\n\n".join([paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text])  # Ensure proper spacing
    elif file.type.startswith("image/"):
        image = Image.open(file)
        text = pytesseract.image_to_string(image)
    else:
        st.error("Unsupported file type.")
    return text

# Analyze case details using both models
def analyze_case(case_details):
    # Generate Bi-GRU model prediction
    embedded_input = np.random.randn(1, 512, 768)  # Replace with actual embeddings
    bi_gru_prediction = bi_gru_model.predict(embedded_input)
    bi_gru_verdict = "Guilty" if bi_gru_prediction[0][0] > 0.5 else "Not Guilty"
    
    # Generate rationale and related laws using LLM
    prompt = f"""
    Provide the rationale and relevant laws to support the predicted verdict: {bi_gru_verdict} as per given below response format having
    Case Details:
    {case_details}

    Response format:
    Rationale: (Short, concise, in future tense and directly to the point)
    Relevant Laws: (List legal references or principles that support the verdict)
    """
    response = mistral_llm(prompt)
    
    # Return the verdict followed by the rationale and relevant laws
    return f"Verdict: {bi_gru_verdict}\n\n{response.strip()}"

# Main section
document_text = ""
uploaded_file = st.file_uploader("Upload Legal Document (PDF, DOCX, or scanned image)")
if uploaded_file:
    document_text = extract_text_from_file(uploaded_file)
case_details = st.text_area("Extracted file for analysis:", document_text, height=300)
if st.button("Analyze Case"):
    verdict_prediction = analyze_case(case_details)
    st.subheader("Predicted Verdict and Analysis")
    st.write(verdict_prediction)
