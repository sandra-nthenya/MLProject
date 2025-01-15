### **Code Overview**

1. **Imports**:
    ```python
    import streamlit as st
    import re
    import pickle
    import numpy as np
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import load_model
    ```

    - **`streamlit as st`**: Streamlit is used to create the interactive web interface. It allows you to build and deploy machine learning applications with minimal code.
    - **`re`**: This is Python’s regular expression module, used to search for patterns in text (such as detecting sarcasm).
    - **`pickle`**: This module is used for serializing and deserializing Python objects. Here, it is used to load the pre-trained tokenizer.
    - **`numpy as np`**: Used for numerical computations. In this code, it’s used to handle predictions.
    - **`pad_sequences`**: A function from TensorFlow's `keras` library, used to ensure that text sequences have consistent lengths by padding shorter sequences.
    - **`load_model`**: This function loads a pre-trained model, which will be used to predict sentiment classes.

---

2. **Loading the Tokenizer and Model**:
    ```python
    with open('/content/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    model = load_model("/content/model.h5")
    ```

    - **Tokenizer**: The tokenizer is loaded from a file (`tokenizer.pkl`) using the `pickle.load()` method. It is used to convert raw text into sequences of integers, which the model can understand.
    - **Model**: The pre-trained model (`model.h5`) is loaded using TensorFlow’s `load_model` function. This model is used for sentiment classification.

---

3. **Text Cleaning Function**:
    ```python
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    ```

    - **Purpose**: This function cleans the input text to ensure it is in a suitable format for prediction.
    - **Steps**:
        - **Lowercase**: Converts the entire text to lowercase for uniformity.
        - **Remove URLs**: Removes any URLs in the text using a regular expression.
        - **Remove HTML Tags**: Removes HTML tags (like `<div>`, `<p>`) using another regular expression.
        - **Remove Special Characters & Numbers**: Keeps only alphabets and spaces by removing other characters.
        - **Extra Spaces**: Condenses multiple spaces into a single space and removes leading/trailing spaces.

---

4. **Prediction Functions**:

    **Sentiment Prediction**:
    ```python
    def predict_classes(sample_text):
        cleaned_sample_text = clean_text(sample_text)
        sample_sequence = tokenizer.texts_to_sequences([cleaned_sample_text])
        sample_padded = pad_sequences(sample_sequence, maxlen=100)
        sample_prediction = model.predict(sample_padded)
        predicted_class = np.argmax(sample_prediction, axis=1)[0]
        class_names = ['Negative', 'Neutral', 'Positive', 'Irrelevant']
        return class_names[predicted_class]
    ```

    - **Text Preprocessing**: The input text is cleaned using the `clean_text()` function.
    - **Tokenization**: The cleaned text is converted into a sequence of integers using the tokenizer.
    - **Padding**: The sequence is padded to ensure a consistent length (100 tokens in this case).
    - **Prediction**: The padded sequence is passed into the model to predict the sentiment. The model outputs probabilities for each class, and the `np.argmax()` function selects the class with the highest probability.
    - **Class Mapping**: The predicted class (an integer) is mapped to a corresponding sentiment label: `Negative`, `Neutral`, `Positive`, or `Irrelevant`.

    **Sarcasm Prediction**:
    ```python
    def predict_sarcasm(text):
        if not isinstance(text, str):
            return "Text entered is not a string"
        text = text.lower().strip()
        sarcasm_patterns = [
            r'\byeah,? right\b', r'\btotally\b', r'\bsure\b', r'\bof course\b',
            r'\bas if\b', r'\bgreat,? just what i needed\b', r'\blove that for me\b',
            r'\bwhat a surprise\b', r'\bthanks a lot\b',
        ]
        punctuation_patterns = [
            r'!{2,}', r'\.{3,}', r'\b(not|never|no way) (really|totally|at all)\b',
        ]
        for pattern in sarcasm_patterns + punctuation_patterns:
            if re.search(pattern, text):
                return "Sarcastic"
        return "Not Sarcastic"
    ```

    - **Sarcasm Detection**: This function checks the input text for specific patterns that are commonly associated with sarcasm.
    - **Pattern Matching**: It uses regular expressions to search for words and phrases like "yeah right", "totally", "sure", etc., and patterns like multiple exclamation marks or ellipses.
    - **Return Value**: If any of the patterns match, it returns `"Sarcastic"`. If no match is found, it returns `"Not Sarcastic"`.

---

5. **Running the Project**
**Streamlit User Interface**:
    ```python
    st.title("Text Analysis: Sentiment and Sarcasm Detection")
    st.write("Enter your text below, and the app will predict its sentiment class and sarcasm.")
    user_input = st.text_area("Input Text", value="", placeholder="Type your text here...")

    if st.button("Analyze"):
        if user_input.strip():
            sentiment_result = predict_classes(user_input)
            sarcasm_result = predict_sarcasm(user_input)
            st.write(f"**Sentiment Class:** {sentiment_result}")
            st.write(f"**Sarcasm Detection:** {sarcasm_result}")
        else:
            st.warning("Please enter some text to analyze.")
    ```

    - **`st.title()`**: Sets the title of the Streamlit app.
    - **`st.write()`**: Displays text on the page to guide the user.
    - **`st.text_area()`**: Creates a text input area where users can type or paste text.
    - **`st.button("Analyze")`**: When clicked, it triggers the analysis of the input text.
    - **Prediction Display**: If the user input is not empty, the app calls both `predict_classes()` and `predict_sarcasm()` functions and displays the results (Sentiment and Sarcasm). If no input is provided, it shows a warning.

---

**OR**

**Web Extension**:
- Run the app.py under extension after the model has been trained and there is a model.h5 file and a tokenizer.pkl file.
- Move the two files into the extension folder.
- Go to chrome extensions, enter developer mode and import the extension folder.
- Load Twitter/X on the chrome brower and clock Activate Sentiment Analysis on the extension.

---

### **Flow of the Application**

1. **User Input**: The user types or pastes text into the provided input area.
2. **Text Analysis**:
   - The app cleans the input text (removes unnecessary characters, URLs, etc.).
   - The text is tokenized and padded, and then passed into the sentiment model to predict its sentiment.
   - The text is checked for sarcasm using predefined regular expressions.
3. **Output**:
   - The app displays the predicted sentiment (e.g., "Positive", "Negative") and sarcasm status ("Sarcastic" or "Not Sarcastic").

---

### **How the Application Works**

1. **Preprocessing**: Text is cleaned to remove noise and make it easier for the model to interpret.
2. **Prediction**: The model predicts the sentiment, and regex patterns detect sarcasm.
3. **Interactive Web Interface**: The Streamlit app provides an easy-to-use interface where users can input text and get predictions instantly.
4. **Real-time Sentiment Analysis on the browser:** The web extension is able to pick up the trained model and run sentiment analysis in real time from a chrome extension.
