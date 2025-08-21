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
 ]

def main():
    #arguments
    parser = argparse.ArgumentParser(description="Classify SMS messages as spam or not-spam using a trained model.")
    parser.add_argument("-m", "--messages", nargs="*", help="Messages to classify (use // to separate multiple messages). If not provided, runs demo messages.")
    args = parser.parse_args()

    #Load model
    model_filename = "spam_detection_model.joblib"
    try:
        loaded_model = joblib.load(model_filename)
    except FileNotFoundError:
        print(f"Error: Model file '{model_filename}' not found. Please ensure the model is in the same directory.")
        return

    if args.messages:
     
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

  
    processed_messages = [preprocess_message(msg) for msg in messages]
  predictions = loaded_model.predict(processed_messages)
    prediction_probabilities = loaded_model.predict_proba(processed_messages)

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
