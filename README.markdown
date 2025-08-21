# ML-SMS-Spam-Classification-Model

Hey there! Welcome to **ML-SMS-Spam-Classification-Model**, This repo holds a Python script that uses a pre-trained machine learning model to classify SMS messages as spam or not-spam (ham). It's built with scikit-learn's Naive Bayes classifier, a sprinkle of NLTK for text preprocessing, and joblib for model persistence. Whether you're testing the demo messages or throwing in your own, this tool's got you covered.



## What's Inside?

The script (`spam_predictor.py`) loads a pre-trained model and classifies SMS messages. It uses a pipeline with a TF-IDF vectorizer and a Multinomial Naive Bayes classifier, fine-tuned for spam detection. You can run it with demo messages or pass your own via the command line using the `-m` flag—no quotes needed!

### Features
- **Spam Detection**: Labels messages as spam or not-spam with probability scores.
- **Text Preprocessing**: Cleans messages with lowercasing, regex, stop word removal, and stemming.
- **Flexible Input**: Use the `-m` flag to classify custom messages (no quotes needed) or stick with the demo set. Use `//` to separate multiple messages.
- **Model Persistence**: Loads a pre-trained model saved with joblib for quick predictions.

## Getting Started

### Prerequisites
You’ll need Python 3.6+ and the following packages:

```bash
pip install -r requirements.txt
```

Also, grab some NLTK data for preprocessing:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

You’ll need the pre-trained model file (`spam_detection_model.joblib`) from this same repo.

### Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/mushrafmustafa/ML-sms-spam-classification-model.git
   cd ML-sms-spam-classification-model
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure `spam_detection_model.joblib` is in the repo directory.

### Usage
Run the script with demo messages (default) or provide your own with the `-m` flag. You don’t need quotes for messages, and you can use `--` to separate multiple messages.

#### Run Demo Messages
```bash
python spam_predictor.py
```

Output example:
```
Message 1: Congratulations! You've won a free trip to Stockholm for 2 people...
Prediction: Spam
Spam Probability: 1.00
Not-Spam Probability: 0.00
--------------------------------------------------
Message 2: Hey, are we still meeting up for lunch today?
Prediction: Not-Spam
Spam Probability: 0.00
Not-Spam Probability: 1.00
--------------------------------------------------
...
```

#### Classify Custom Messages
Pass a message with `-m` (no quotes needed):
```bash
python spam_predictor.py -m Congratulations! You've won a free trip to Stockholm for 2 people.
```

Output example:
```
Message 1: Congratulations! You've won a free trip to Stockholm for 2 people.
Prediction: Spam
Spam Probability: 0.98
Not-Spam Probability: 0.02
--------------------------------------------------
```

For multiple messages, use `//` to separate them:
```bash
python spam_predictor.py -m Win a free Volvo car now! // Lunch at 1pm?
```

Output example:
```
Message 1: Win a free iPhone now!
Prediction: Spam
Spam Probability: 0.98
Not-Spam Probability: 0.02
--------------------------------------------------
Message 2: Lunch at 1pm?
Prediction: Not-Spam
Spam Probability: 0.01
Not-Spam Probability: 0.99
--------------------------------------------------
```

## How It Works
1. **Preprocessing**: Messages are lowercased, cleaned with regex, tokenized, stripped of stop words, and stemmed to match the training data format.
2. **Model**: The script loads a pre-trained pipeline (TF-IDF vectorizer + Multinomial Naive Bayes) from `spam_detection_model.joblib`.
3. **Prediction**: Processes input messages (demo or user-provided) and outputs predictions with spam/not-spam probabilities.

The model was trained and tuned in the [ML-sms-spam-classification-model](https://github.com/mushrafmustafa/ML-sms-spam-classification-model) repo using GridSearchCV to optimize the Naive Bayes `alpha` parameter.

## Gotchas
- **Model File**: Make sure `spam_detection_model.joblib` is in the directory, or you’ll get a FileNotFoundError.
- **Preprocessing**: The script preprocesses messages exactly as done during training. If your training setup differs, predictions may vary.
- **Dependencies**: Ensure NLTK data is downloaded, or preprocessing will fail.

## Future Ideas
- Deploy as a web API with Flask or FastAPI.
- Add support for batch processing from a file.
- Add a GUI for non-technical users.

## Contributing
Got ideas to make this better? Fork the repo, tweak the code, or open an issue! Pull requests are super welcome.

## License
MIT License. Use it, share it, just don’t spam me about it 😄.

## Shoutouts
- Built with scikit-learn, NLTK, and joblib.
- Inspired by the endless battle against spam texts.
- Thanks to the ML community for awesome tools and datasets!

Questions? Bugs? Hit me up on GitHub or open an issue. Happy spam slaying!