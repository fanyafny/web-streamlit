import streamlit as st
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from huggingface_hub import hf_hub_download

# Limit TensorFlow memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load the pre-trained BERT model and tokenizer
PRE_TRAINED_MODEL = 'indobenchmark/indobert-base-p2'
bert_model = TFBertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL, num_labels=2)

# Download the model weights from Hugging Face
model_path = hf_hub_download(repo_id="JokiTugasCoding/bert-model", filename="bert-model.h5")
bert_model.load_weights(model_path)

bert_tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)

# Define the maximum length for padding/truncating sequences
MAX_LEN = 30

# Function to preprocess and tokenize the input text
def preprocess_and_tokenize(text):
    encoded_input = bert_tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='tf'
    )
    return encoded_input['input_ids'], encoded_input['attention_mask'], encoded_input['token_type_ids']

# Function to make a prediction
def predict_clickbait(headline):
    input_ids, attention_mask, token_type_ids = preprocess_and_tokenize(headline)
    prediction = bert_model([input_ids, attention_mask, token_type_ids], training=False)
    predicted_label = tf.argmax(prediction.logits, axis=1).numpy()[0]
    return 'Clickbait' if predicted_label == 1 else 'Bukan Clickbait'

# Streamlit application layout
st.title("Prediksi Clickbait Headline")
st.write("Masukkan Teks Headline")

# Input text from user
headline = st.text_input("")

# Button for making prediction
if st.button("Hasil Deteksi", key='detect_button', use_container_width=True, disabled=False):
    if headline:
        result = predict_clickbait(headline)
        if result == 'Clickbait':
            st.markdown(f"<div style='background-color: lightcoral; color: red; padding: 10px;'>{result}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color: lightgreen; color: darkgreen; padding: 10px;'>{result}</div>", unsafe_allow_html=True)
    else:
        st.warning("Masukkan teks headline terlebih dahulu.")

