from transformers import pipeline
import pandas as pd

def load_model():
    print("Loading BERT sentiment model...")
    classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    print("Model loaded!")
    return classifier

def analyze_sentiment(df, classifier):
    print(f"Analyzing {len(df)} posts...")
    
    results = []
    for text in df['text']:
        result = classifier(text[:512])[0]
        results.append({
            'label': result['label'],
            'score': round(result['score'], 4)
        })
    
    df['sentiment'] = [r['label'] for r in results]
    df['confidence'] = [r['score'] for r in results]
    
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/posts.csv")
    classifier = load_model()
    df = analyze_sentiment(df, classifier)
    df.to_csv("output/sentiment_results.csv", index=False)
    print("Done! Results saved to output/sentiment_results.csv")
    print(df['sentiment'].value_counts())