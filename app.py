import re
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler

class NSFWData:
    def __init__(self, model_file='nsfw_model.pkl', vectorizer_file='vectorizer.pkl'):
        self.model_file = model_file
        self.vectorizer_file = vectorizer_file
        self.spam_patterns = {}
        self.model = LogisticRegression()
        self.vectorizer = None
        self.data = pd.DataFrame(columns=['text', 'label'])  # Empty DataFrame for storing new data

    def create_regex(self, word):
        """Automatically creates a regex pattern for the given spam/aggressive word."""
        pattern = r''.join([f'{char}[\s\*\_\-]?' for char in word])
        return pattern

    def preprocess_text(self, text):
        """Preprocess the text to match spam/aggressive word patterns."""
        for word, pattern in self.spam_patterns.items():
            text = re.sub(pattern, word, text, flags=re.IGNORECASE)
        return text

    def add_data(self, word, message, label):
        """
        Add new spam/non-spam data.
        :param word: The word (spam/aggressive) to be added.
        :param message: The message that contains the word.
        :param label: 1 for spam/aggressive, 0 for non-spam.
        """
        # Automatically create a regex pattern for the spam word
        if label == 1 and word not in self.spam_patterns:
            self.spam_patterns[word] = self.create_regex(word)
            print(f"Added new spam word: '{word}' with regex: {self.spam_patterns[word]}")

        # Preprocess the message to replace the word with its base version (if it's a spam message)
        processed_message = self.preprocess_text(message)

        # Add the new message to the dataset with its label (spam=1, non-spam=0)
        new_data = pd.DataFrame({'text': [message], 'label': [label], 'processed_text': [processed_message]})

        # Append new data to the existing dataset
        self.data = pd.concat([self.data, new_data], ignore_index=True)

        # Only fit the vectorizer after all data has been added
        self.vectorizer = CountVectorizer(lowercase=True, binary=True)

        # Fit and transform the text data
        X = self.vectorizer.fit_transform(self.data['processed_text'])
        y = self.data['label'].astype(int)  # Ensure labels are integers

        # Check if both classes are present
        if len(set(y)) < 2:
            print("Error: Dataset needs to have both spam (1) and non-spam (0) examples.")
            return

        # Resample to handle class imbalance
        oversample = RandomOverSampler(sampling_strategy='minority')
        X_resampled, y_resampled = oversample.fit_resample(X, y)

        # Train the model
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)
        self.model.fit(X_train, y_train)

        # Save the model and vectorizer to disk
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.model, f)
        with open(self.vectorizer_file, 'wb') as f:
            pickle.dump(self.vectorizer, f)

        # Print classification report for evaluation
        y_pred = self.model.predict(X_test)
        print("\nClassification Report (Evaluation on test set):")
        print(classification_report(y_test, y_pred))

    def load_model(self):
        """Load the model and vectorizer from disk."""
        try:
            with open(self.model_file, 'rb') as f:
                self.model = pickle.load(f)
            with open(self.vectorizer_file, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print("Model and vectorizer loaded successfully.")
        except FileNotFoundError:
            print("No model found, please add data first.")

    def check_message(self, sentence):
        """Classify a given message."""
        processed_sentence = self.preprocess_text(sentence)
        sentence_vector = self.vectorizer.transform([processed_sentence])

        # Predict the class (1 = Spam/Aggressive, 0 = Non-Spam/Non-Aggressive)
        prediction = self.model.predict(sentence_vector)

        return "Spam/Aggressive" if prediction[0] == 1 else "Non-Spam/Non-Aggressive"


# Client Code Example

# Initialize the NSFWData class
nsfw_model = NSFWData()

# Adding spam and non-spam data
nsfw_model.add_data("spam", "This is a s*p*a*m message", label=1)
nsfw_model.add_data("buy", "b*u_y now and get 50% off", label=1)
nsfw_model.add_data("idiot", "You are an id_i*o_t", label=1)

# Adding non-spam data
nsfw_model.add_data("", "Hello, how are you?", label=0)
nsfw_model.add_data("", "Let’s catch up later.", label=0)

# Test the model with a new message
while True:
    new_message = input("You:")
    result = nsfw_model.check_message(new_message)
    print(f"\nPrediction for '{new_message}': {result}")

# Optionally, you can load the saved model and test again
# nsfw_model.load_model()
