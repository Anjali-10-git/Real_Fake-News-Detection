# Real_Fake-News-Detection

 Try Fake News Detector: https://realfake-news-detection-uq6rhkat8hffynk5j3ctcz.streamlit.app/

# 📰 Fake News Detector

A simple Streamlit web app that estimates whether a news article is **real** or **fake** based on its writing style.

## Features

- Paste an article and get a verdict with a confidence score
- Fake vs. real probability bar
- Shows the words that influenced the result
- One-click sample articles to try it out

## How it works

The model is a scikit-learn pipeline: **TF-IDF** features feeding a **Logistic Regression** classifier. Text is cleaned by the `wordopt` function in `app.py` before it is vectorized.

> ⚠️ The app judges writing patterns, not facts. It is not a fact-checker, so always verify important claims with a trusted source.
