# EmotionalAI
### By Amit Joshi

EmotionalAI is a machine learning model that classifies the given text to positive and negative sentiment.

<img src="src/static/img/sentiment.png?raw=true" width="1000">

### How to Run:
1. Install necessary modules with `sudo pip3 install -r requirements.txt` command.
2. Go to __src__ folder (if you want to change paths of files and folders, go to _**src/config.py**_).
3. Run `python3 train.py` to train and save the machine learning model.
4. Now Run `python3 app.py` and go to **http://127.0.0.1:5000/** in your browser.
5. You will see the home page click the __Let's Go__ button (or got to __http://127.0.0.1:5000/sentiment.html__) to go to sentiment classifier page.
6. Type in your text and hit the __Get Results__ button.

## Inside the model
* Dataset is provided in the __*data*__ folder, which has been downloaded from [here](https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv)
* The features are extracted using [Distilbert](https://huggingface.co/transformers/model_doc/distilbert.html) from huggingface which is a smaller distilled version of bert.
* Machine Learning model used was SVM.
* Frontend made using Html, CSS and a little bit of JavaScript.
* Backend made using Flask.

__Please Give a :star2: if you :+1: it.__
