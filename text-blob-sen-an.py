from textblob import TextBlob


texts = ["Kalki movie is great", "The capital of India is Delhi"]

if __name__ == "__main__":
    print("Running as main script")
    for text in texts:
        print(text)
        print(f"Polarity | {TextBlob(text).sentiment.polarity}")
        print(f"Subjectivity | {TextBlob(text).sentiment.subjectivity}")
        print("="*50)
