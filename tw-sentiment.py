from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

tweet = "@asleshsai2003 It's a new project   ☺ https://aslesh2003.com"


#classify the words in the sentence 
tweet_words = []

for word in tweet.split(' '):
    if word.startswith('@') and len(word) > 1:
        word = '@user'
    
    elif word.startswith('http'):
        word = "http"
    tweet_words.append(word)

tweet_proc = " ".join(tweet_words)

# load the  model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative', 'Neutral', 'Positive']

# now run sentiment analysis
# use softax to get propability distribution
encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
output = model(**encoded_tweet)
print(output)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
for i in range(len(scores)):
    l = labels[i]
    s = scores[i]
    print(l,s)



