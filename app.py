import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler

# Step 1: Prepare the training dataset
data = {'text': ['This is a spam message', 'Hello friend', 'Buy now and get 50% off', 'You are an idiot',
                 'Check this out, free gift', 'Amazing offer', 'Hello, how are you', 'Let’s catch up soon'],
        'label': [1, 0, 1, 1, 1, 1, 0, 0]}  # Imbalanced dataset

df = pd.DataFrame(data)

# Step 2: Define the custom spam/aggressive words list and their regex patterns
spam_patterns = {
    'spam': r's\s*p\s*a\s*m',
    'buy': r'b[\s\*\_\-]?u[\s\*\_\-]?y',  # buy = b u y, bu y, b*y, b_u_y, etc.
    'idiot': r'i[\s\*\_\-]?d[\s\*\_\-]?i[\s\*\_\-]?o[\s\*\_\-]?t',
    '50% off': r'50[\s\*\_\-]?%[\s\*\_\-]?off',
    'free': r'f[\s\*\_\-]?r[\s\*\_\-]?e[\s\*\_\-]?e',
    'gift': r'g[\s\*\_\-]?i[\s\*\_\-]?f[\s\*\_\-]?t',
    'offer': r'o[\s\*\_\-]?f[\s\*\_\-]?f[\s\*\_\-]?e[\s\*\_\-]?r'
}

# Step 3: Preprocess the text to match the regex patterns
def preprocess_text(text):
    for word, pattern in spam_patterns.items():
        # Replace all occurrences of pattern with the word to make detection easier
        text = re.sub(pattern, word, text, flags=re.IGNORECASE)
    return text

# Apply the preprocessing function to the dataset
df['processed_text'] = df['text'].apply(preprocess_text)

# Step 4: Vectorization (Bag of Words)
vectorizer = CountVectorizer(vocabulary=spam_patterns.keys(), lowercase=True, binary=True)
X = vectorizer.fit_transform(df['processed_text'])
y = df['label']

# Step 5: Handle Imbalanced Data with Random Oversampling
oversample = RandomOverSampler(sampling_strategy='minority')
X_resampled, y_resampled = oversample.fit_resample(X, y)

# Step 6: Train-Test Split (Stratified)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Step 7: Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 8: Function to Test the Model on New Sentences
def test_model(sentence):
    # Preprocess and vectorize the new sentence
    processed_sentence = preprocess_text(sentence)
    sentence_vector = vectorizer.transform([processed_sentence])
    
    # Predict the class (1 = Spam/Aggressive, 0 = Non-Spam/Non-Aggressive)
    prediction = model.predict(sentence_vector)
    
    # Output prediction result
    if prediction[0] == 1:
        return "Spam/Aggressive"
    else:
        return "Non-Spam/Non-Aggressive"

# Step 9: Client to Test Model with User Input
while True:
    user_input = input("\nEnter a sentence to test (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    result = test_model(user_input)
    print(f"Prediction: {result}")

# Step 10: Evaluate the model using the test set
print("\nClassification Report (Evaluation on test set):")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Step 11: Show the words the model is looking for
print("\nSpam/aggressive words and patterns being detected:")
for word, pattern in spam_patterns.items():
    print(f"{word}: {pattern}")
