import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Sample Tweets
data = {
    "tweets": [
        "I love this phone!",
        "This movie is terrible.",
        "What a fantastic day!",
        "I'm so tired of this traffic.",
        "The service was okay, nothing special.",
        "I am extremely happy with the results.",
        "I hate waiting in long lines!",
        "Feeling great after the workout!",
        "This is the worst product ever.",
        "Not bad, but could be better."
    ]
}

df = pd.DataFrame(data)

# Sentiment Analysis Function
def analyze_sentiment(tweet):
    blob = TextBlob(tweet)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Apply Analysis
df["Sentiment"] = df["tweets"].apply(analyze_sentiment)
print(df)

# Visualize Results
df["Sentiment"].value_counts().plot(kind='bar', color=['green', 'red', 'gray'])
plt.title("Sentiment Analysis of Tweets")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.grid(True)
plt.show()

