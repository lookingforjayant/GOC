from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

texts = ["Kalki movie is great", "The capital of India is Delhi"]


if __name__ == "__main__":
    print("Running as main script")
    sentiment_analyzer = SentimentIntensityAnalyzer()
    for text in texts:
        print(f"Polarity | {sentiment_analyzer.polarity_scores(text)}")
        print("="*50)
