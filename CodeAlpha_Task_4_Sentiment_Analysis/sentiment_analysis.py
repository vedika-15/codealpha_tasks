import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob

# Load the dataset
df = pd.read_csv("product_reviews.csv")

# Function to analyze sentiment
def analyze_sentiment(text):
    if pd.isna(text):   # handle missing values
        return "Neutral"
    
    polarity = TextBlob(str(text)).sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"


# Apply sentiment analysis
df["Sentiment"] = df["Review"].apply(analyze_sentiment)

# Display sentiment count
sentiment_counts = df["Sentiment"].value_counts()
print("Sentiment Distribution:")
print(sentiment_counts)

# -------------------- BAR CHART --------------------
plt.figure(figsize=(6,4))
plt.bar(sentiment_counts.index, sentiment_counts.values)
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.title("Sentiment Analysis of Product Reviews")
plt.show()

# -------------------- PIE CHART --------------------
plt.figure(figsize=(6,6))
plt.pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct="%1.1f%%",
    startangle=140
)
plt.title("Sentiment Percentage Distribution")
plt.show()

# -------------------- SAMPLE REVIEWS --------------------
print("\nTop Positive Reviews:")
positive_reviews = df[df["Sentiment"] == "Positive"]["Review"].head(5)
for review in positive_reviews:
    print("-", review)

print("\nTop Negative Reviews:")
negative_reviews = df[df["Sentiment"] == "Negative"]["Review"].head(5)
for review in negative_reviews:
    print("-", review)

# -------------------- SAVE RESULTS --------------------
df.to_csv("sentiment_results.csv", index=False)
print("\nSentiment analysis results saved to sentiment_results.csv")

