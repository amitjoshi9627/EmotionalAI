from flask import Flask, redirect, url_for, render_template, request, jsonify
from sentiment_classifier import get_prediction
import time
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/sentiment.html")
def sentiment_classifier():
    return render_template("sentiment.html")


@app.route("/background_process")
def background_process():
    try:
        text = request.args.get('textdata')
        result = get_prediction(text)
        if result:
            prediction = "Positive Sentiment"
        else:
            prediction = "Negative Sentiment"
        return jsonify(prediction=prediction)
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(debug=True)
