# YoutubeSpamDetector

A PyTorch and Streamlit web app that detects spam-like YouTube comments. Users can enter a single comment or upload a CSV file to classify multiple comments, view spam probabilities, and see basic prediction visualizations.

## Features

* Classifies YouTube-style comments as spam or not spam
* Shows spam probability for each prediction
* Supports single-comment input
* Supports CSV upload with a `text` column
* Displays prediction results in a table
* Shows a simple bar chart of spam vs normal comments
* Built with a custom PyTorch neural network
* Deployed with Streamlit

## Tech Stack

* Python
* PyTorch
* Streamlit
* Pandas
* Matplotlib
* Scikit-learn

## How It Works

The app uses a basic bag-of-words text classification approach.

1. The text is cleaned and converted to lowercase.
2. The comment is split into words.
3. Words are converted into a numeric vector using a vocabulary built from the dataset.
4. A small PyTorch neural network predicts whether the comment is spam.
5. Streamlit displays the prediction and spam probability.

## Project Structure

```text
YoutubeSpamDetector/
├── app.py
├── model.py
├── train.py
├── data.csv
├── spam_model.pt
├── requirements.txt
├── .gitignore
└── README.md
```

## Files

* `app.py` — Streamlit web app
* `model.py` — PyTorch neural network model
* `train.py` — training script for the spam classifier
* `data.csv` — training dataset
* `spam_model.pt` — saved trained PyTorch model
* `requirements.txt` — project dependencies

## Run Locally

Clone the repository:

```bash
git clone https://github.com/FJamal1200/YoutubeSpamDetector.git
cd YoutubeSpamDetector
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python train.py
```

Run the Streamlit app:

```bash
python -m streamlit run app.py
```

Then open the local Streamlit URL in your browser.

## CSV Format

To analyze multiple comments, upload a CSV file with a column named `text`.

Example:

```csv
text
"great video bro"
"check out my channel"
"free crypto giveaway click here"
```

## Model

The model is a simple feed-forward PyTorch neural network:

```text
Bag-of-words vector → Linear layer → ReLU → Linear layer → Spam probability
```

This project is meant as a beginner-friendly data science and PyTorch deployment project, not a production-level moderation system.

## Future Improvements

* Add more training data
* Improve text preprocessing
* Add support for more labels like toxic, useful feedback, and self-promotion
* Add model accuracy metrics
* Add confusion matrix and evaluation charts
* Try a more advanced NLP model later

