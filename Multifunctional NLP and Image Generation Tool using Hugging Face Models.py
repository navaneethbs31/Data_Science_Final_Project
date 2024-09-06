import streamlit as st
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
from transformers import CLIPProcessor, CLIPModel

# Load Models
summarizer = pipeline("summarization")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")
sentiment_analyzer = pipeline("sentiment-analysis")
qa = pipeline("question-answering")
# image_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32") # Placeholder for image generation model
# image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Functions for tasks
def summarize_text(text):
    return summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']

def predict_next_word(text):
    inputs = tokenizer.encode(text, return_tensors="pt")
    outputs = gpt2_model.generate(inputs, max_length=50, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_story(prompt):
    return predict_next_word(prompt)

def chat_with_bot(text):
    response = chatbot(text)[0]
    return response['generated_text']

def analyze_sentiment(text):
    return sentiment_analyzer(text)[0]

def answer_question(question, context):
    return qa(question=question, context=context)['answer']

# Placeholder for image generation function
# def generate_image(prompt):
#     inputs = image_processor(text=prompt, return_tensors="pt")
#     image = image_model.generate(inputs)
#     return image

# Streamlit Interface
st.title("Multifunctional NLP and Image Generation Tool")

task = st.sidebar.selectbox(
    "Choose a Task",
    ["Text Summarization", "Next Word Prediction", "Story Prediction",
     "Chatbot", "Sentiment Analysis", "Question Answering", "Image Generation"]
)

if task == "Text Summarization":
    text = st.text_area("Enter text for summarization:")
    if st.button("Summarize"):
        summary = summarize_text(text)
        st.write("Summary:")
        st.write(summary)

elif task == "Next Word Prediction":
    text = st.text_area("Enter text for next word prediction:")
    if st.button("Predict Next Word"):
        prediction = predict_next_word(text)
        st.write("Prediction:")
        st.write(prediction)

elif task == "Story Prediction":
    text = st.text_area("Enter a prompt for story prediction:")
    if st.button("Generate Story"):
        story = generate_story(text)
        st.write("Generated Story:")
        st.write(story)

elif task == "Chatbot":
    text = st.text_area("Chat with the bot:")
    if st.button("Send"):
        response = chat_with_bot(text)
        st.write("Bot Response:")
        st.write(response)

elif task == "Sentiment Analysis":
    text = st.text_area("Enter text for sentiment analysis:")
    if st.button("Analyze Sentiment"):
        sentiment = analyze_sentiment(text)
        st.write("Sentiment:")
        st.write(f"Label: {sentiment['label']}, Score: {sentiment['score']}")

elif task == "Question Answering":
    question = st.text_input("Enter your question:")
    context = st.text_area("Enter the context:")
    if st.button("Get Answer"):
        answer = answer_question(question, context)
        st.write("Answer:")
        st.write(answer)

elif task == "Image Generation":
    prompt = st.text_area("Enter a prompt for image generation:")
    if st.button("Generate Image"):
        st.write("Image generation feature is currently a placeholder.")
        # Uncomment the following lines to enable image generation
        # image = generate_image(prompt)
        # st.image(image)

st.sidebar.write("This tool uses various pretrained models from Hugging Face.")
