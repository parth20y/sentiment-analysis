import pandas as pd
from sentiment_model import load_model, analyze_sentiment
import os

def run_pipeline():
    print("=== Sentiment Analysis Pipeline ===")
    
    # Step 1 — Load data
    print("\n[1/3] Loading data...")
    df = pd.read_csv("data/posts.csv")
    print(f"Loaded {len(df)} posts")

    # Step 2 — Run sentiment analysis
    print("\n[2/3] Running BERT sentiment analysis...")
    classifier = load_model()
    df = analyze_sentiment(df, classifier)

    # Step 3 — Save enriched output
    print("\n[3/3] Saving results...")
    os.makedirs("output", exist_ok=True)
    df.to_csv("output/sentiment_results.csv", index=False)

    # Summary stats
    print("\n=== Summary ===")
    print(f"Total posts processed: {len(df)}")
    print(f"Sentiment breakdown:\n{df['sentiment'].value_counts()}")
    print(f"Average confidence: {df['confidence'].mean():.4f}")
    print(f"\nBy subreddit:")
    print(df.groupby(['subreddit','sentiment']).size().unstack(fill_value=0))
    
    print("\nPipeline complete! Check output/sentiment_results.csv")

if __name__ == "__main__":
    run_pipeline()