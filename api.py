from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS #CORS(app) is used to enable CORS for the entire application.
import re
from io import BytesIO

# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import base64

STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)
CORS(app)

@app.route("/test", methods=["GET"])
def test():
    return "Test request received successfully. Service is running."


@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("landing.html")


@app.route("/predict", methods=["POST"])
def predict():
    # Select the predictor to be loaded from Models folder
    # predictor is the trained model (in this case, an XGBoost model, since it had the highest testing accuracy).
    predictor = pickle.load(open(r"Models/model_xgb.pkl", "rb"))

    # scaler is used to scale the data to a standard range.
    scaler = pickle.load(open(r"Models/scaler.pkl", "rb"))

    # cv is the CountVectorizer, which converts text into numerical features.
    cv = pickle.load(open(r"Models/countVectorizer.pkl", "rb"))

    try:
        # Check if the request contains a file (for bulk prediction) or text input
        if "file" in request.files:
            
            # Bulk prediction from CSV file
            file = request.files["file"]
            data = pd.read_csv(file)

            predictions, graph = bulk_prediction(predictor, scaler, cv, data)

            # The predictions are sent back to the user as a downloadable CSV file.
            # Headers are added to the response to indicate that a graph exists and include the graph data encoded in base64.
            response = send_file(
                predictions,
                mimetype="text/csv",
                as_attachment=True,
                download_name="Predictions.csv",
            )

            response.headers["X-Graph-Exists"] = "true"

            response.headers["X-Graph-Data"] = base64.b64encode(
                graph.getbuffer()
            ).decode("ascii")

            return response

        # If the request contains a text input instead of a file, the function processes it as a single string prediction.
        # The text is extracted from the request.
        # The function single_prediction is called with the model, scaler, CountVectorizer, and the text. 
        # This function returns the predicted sentiment.
        # The prediction is returned as a JSON response.
        elif "text" in request.json:
            # Single string prediction
            text_input = request.json["text"]
            predicted_sentiment = single_prediction(predictor, scaler, cv, text_input)

            return jsonify({"prediction": predicted_sentiment})

    # If any error occurs during the process, it is caught and returned as a JSON response containing the error message.
    except Exception as e:
        return jsonify({"error": str(e)})


def single_prediction(predictor, scaler, cv, text_input):
    # corpus: an empty list to store the processed text.
    # stemmer: an instance of PorterStemmer used for stemming words (reducing words to their root form).
    corpus = []
    stemmer = PorterStemmer()

    # This line uses a regular expression (re.sub) to replace all characters in text_input 
    # that are not letters (a-z, A-Z) with a space. 
    # This helps to remove punctuation and numbers, leaving only letters.
    review = re.sub("[^a-zA-Z]", " ", text_input)

    # This converts the text to lowercase and splits it into individual words. 
    # Lowercasing helps to ensure that words like "Apple" and "apple" are treated the same.
    review = review.lower().split()

    # This line performs two tasks:
    # Stemming: Each word in the list is stemmed using stemmer.stem(word).
    # Stopwords Removal: It excludes words that are in the STOPWORDS list (common words like "the", "is", "in" that don't add much meaning).
    review = [stemmer.stem(word) for word in review if not word in STOPWORDS]

    #This joins the list of stemmed words back into a single string, with words separated by spaces.
    review = " ".join(review)

    #This appends the cleaned and processed text to the corpus lis
    corpus.append(review)

    # This line uses the CountVectorizer (cv) to transform the processed text in corpus into a numerical feature array. 
    # The result, X_prediction, is a numerical representation of the text.
    X_prediction = cv.transform(corpus).toarray()

    # This line uses the scaler to normalize the numerical features in X_prediction, resulting in X_prediction_scl. 
    # Scaling helps to ensure that all features contribute equally to the model's predictions.
    X_prediction_scl = scaler.transform(X_prediction)

    # This line uses the pre-trained model (predictor, in our case here, XGBoost model) to predict the probabilities of the classes 
    # for the input features X_prediction_scl. 
    # The result, y_predictions, is an array of probabilities.
    y_predictions = predictor.predict_proba(X_prediction_scl)

    # This line finds the class with the highest probability. argmax(axis=1) returns the index of the highest probability class 
    # for each sample. 
    # [0] extracts the class index for the single sample.
    y_predictions = y_predictions.argmax(axis=1)[0]

    return "Positive" if y_predictions == 1 else "Negative"

# Steps followed:
# Text Transformation: Convert text data to a numerical format.
# Data Scaling: Scale the numerical data for consistent model performance.
# Probability Prediction: Predict class probabilities using the trained model.
# Class Prediction: Find the class with the highest probability.
# Sentiment Mapping: Map predicted classes to human-readable sentiment labels.
# Update DataFrame: Add predictions to the DataFrame.
# CSV Creation: Create an in-memory CSV file with the predictions.
# Graph Generation: Generate a distribution graph of the predictions.

def bulk_prediction(predictor, scaler, cv, data):
    corpus = []
    stemmer = PorterStemmer()
    for i in range(0, data.shape[0]):
        review = re.sub("[^a-zA-Z]", " ", data.iloc[i]["Sentence"])
        review = review.lower().split()
        review = [stemmer.stem(word) for word in review if not word in STOPWORDS]
        review = " ".join(review)
        corpus.append(review)

    X_prediction = cv.transform(corpus).toarray()
    X_prediction_scl = scaler.transform(X_prediction)

    # This returns a matrix where each row corresponds to a sample, 
    # and each column corresponds to the probability of that sample belonging to a specific class ( class in our context beting 'positive', 'negative' or 'neutral').
    y_predictions = predictor.predict_proba(X_prediction_scl)

    # Following code helps find the index of the class with the highest probability for each sample. 
    # This gives us the predicted class (sentiment) for each sample.
    y_predictions = y_predictions.argmax(axis=1)

    # the following line of code maps the predicted class indices to their corresponding sentiment labels using the sentiment_mapping function. 
    # This converts the numerical class predictions into human-readable sentiment labels (e.g., positive, negative).
    y_predictions = list(map(sentiment_mapping, y_predictions))

    #This line adds the predicted sentiments to the original DataFrame (data) as a new column called "Predicted sentiment".
    data["Predicted sentiment"] = y_predictions

    # BytesIO() creates an in-memory binary stream to hold the CSV data.
    predictions_csv = BytesIO()

    data.to_csv(predictions_csv, index=False)

    #predictions_csv.seek(0) moves the cursor to the beginning of the stream so it can be read from the start.
    predictions_csv.seek(0)

    graph = get_distribution_graph(data)

    return predictions_csv, graph


def get_distribution_graph(data):
    fig = plt.figure(figsize=(5, 5))
    colors = ("green", "red")
    wp = {"linewidth": 1, "edgecolor": "black"}
    tags = data["Predicted sentiment"].value_counts()
    explode = (0.01, 0.01)

    tags.plot(
        kind="pie",
        autopct="%1.1f%%",
        shadow=True,
        colors=colors,
        startangle=90,
        wedgeprops=wp,
        explode=explode,
        title="Sentiment Distribution",
        xlabel="",
        ylabel="",
    )

    graph = BytesIO()
    plt.savefig(graph, format="png")
    plt.close()

    return graph


def sentiment_mapping(x):
    if x == 0:
        return "Negative"    
    else :
        return "Positive"


if __name__ == "__main__":
    app.run(port=5000, debug=True)