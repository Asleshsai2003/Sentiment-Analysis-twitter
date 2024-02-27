# Sentiment Analysis with Twitter-Roberta

This script performs sentiment analysis on a given tweet using the Twitter-Roberta model. It classifies the sentiment as Negative, Neutral, or Positive.

## Requirements

- Python 3.x
- [Transformers library](https://github.com/huggingface/transformers)
- [Scipy library](https://www.scipy.org/)

## Installation

1. Clone the repository:

   
   git clone https://github.com/yourusername/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
Install the required libraries:
2. Download the neccessary libraries


pip install transformers scipy
Run the script:


python sentiment_analysis.py
Usage
The sentiment_analysis.py script takes a tweet as input and classifies its sentiment.


python sentiment_analysis.py
Example:


python sentiment_analysis.py "@asleshsai2003 It's a new project ☺ https://aslesh2003.com"
Output
The script outputs the sentiment probabilities for Negative, Neutral, and Positive.

Example:

Negative 0.043
Neutral 0.367
Positive 0.59
