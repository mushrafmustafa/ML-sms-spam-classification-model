import joblib
import re
import argparse
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize NLTK components
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Preprocess function to mirror training-time preprocessing
def preprocess_message(message):
    message = message.lower()
    message = re.sub(r"[^a-z\s$!]", "", message)
    tokens = word_tokenize(message)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

# Demo messages for default execution
DEMO_MESSAGES = [
    "Congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/1234 to claim now.",
    "Hey, are we still meeting up for lunch today?",
    "Urgent! Your account has been compromised. Verify your details here: www.fakebank.com/verify",
    "Reminder: Your appointment is scheduled for tomorrow at 10am.",
    "Howdy! How r u? How is your day? Please let me know about you personally. I hope you won't be against that :)) I am blond. My eyes 're blue. I 'm hundred sixty-eight centimeters tall. I 'm slender. I am very good-natured and I love to speak to interesting people. I have a great deal of acquaintances but still I prefer spending time with me close pals.Occasionally. I 'm really sentimental and I think that I 'm a hopeful lady. Write me on my email address: Aidan068a@hotmail.com",
    "Urgent! Your account has been compromised. Verify your details here: www.fakebank.com/verify",
    "Reminder: Your appointment is scheduled for tomorrow at 10am.",
]

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Classify SMS messages as spam or not-spam using a trained model.")
    parser.add_argument("-m", "--messages", nargs="*", help="Messages to classify (use // to separate multiple messages). If not provided, runs demo messages.")
    args = parser.parse_args()

    # Load the trained model
    model_filename = "spam_detection_model.joblib"
    try:
        loaded_model = joblib.load(model_filename)
    except FileNotFoundError:
        print(f"Error: Model file '{model_filename}' not found. Please ensure the model is in the same directory.")
        return

    # Determine messages to process
    if args.messages:
        # Join arguments until a separator (//) is found, or take all as one message
        messages = []
        current_message = []
        for arg in args.messages:
            if arg == "//":
                if current_message:
                    messages.append(" ".join(current_message))
                    current_message = []
            else:
                current_message.append(arg)
        if current_message:
            messages.append(" ".join(current_message))
    else:
        messages = DEMO_MESSAGES

    # Preprocess all messages
    processed_messages = [preprocess_message(msg) for msg in messages]

    # Make predictions and get probabilities
    predictions = loaded_model.predict(processed_messages)
    prediction_probabilities = loaded_model.predict_proba(processed_messages)

    # Display predictions
    for i, msg in enumerate(messages):
        prediction = "Spam" if predictions[i] == 1 else "Not-Spam"
        spam_probability = prediction_probabilities[i][1]
        ham_probability = prediction_probabilities[i][0]
        
        print(f"Message {i + 1}: {msg}")
        print(f"Prediction: {prediction}")
        print(f"Spam Probability: {spam_probability:.2f}")
        print(f"Not-Spam Probability: {ham_probability:.2f}")
        print("-" * 50)

if __name__ == "__main__":
    main()