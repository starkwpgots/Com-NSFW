import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas

app = Flask(__name__)

# load data
data = pandas.read_csv('spam.csv', encoding='latin-1')
train_data = data[:4457] # 4457 items 80%
test_data = data[4457:] # 1115 items 20%

# train model
Classifier = RandomForestClassifier(n_estimators=100, n_jobs=1)
Vectorizer = CountVectorizer()
vectorize_text = Vectorizer.fit_transform(train_data.v2)
Classifier.fit(vectorize_text, train_data.v1)

# score
vectorize_text_test = Vectorizer.transform(test_data.v2)
score = Classifier.score(vectorize_text_test, test_data.v1)
print('Score ' + str(score)) # 0.975784753363

@app.route('/', methods=['GET'])
def index():
    message = request.args.get('message', '')
    error = ''
    predict_probability = ''
    predict = ''

    global Classifier
    global Vectorizer

    try:
        if len(message) > 0:
            vectorize_message = Vectorizer.transform([message])
            predict = Classifier.predict(vectorize_message)[0]
            predict_probability = Classifier.predict_proba(vectorize_message).tolist()
    except BaseException as inst:
        error = str(type(inst).__name__) + ' ' + str(inst)
    return jsonify(
        message=message, predict_proba=predict_probability, predict=predict, error=error)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=True)
