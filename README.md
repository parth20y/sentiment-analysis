# Real-Time Sentiment Analysis Dashboard

A BERT-based sentiment analysis pipeline that processes social media posts 
and visualizes results through a Power BI dashboard.

## Project Structure
```
sentiment-analysis/
├── data/
│   └── posts.csv
├── output/
│   └── sentiment_results.csv
├── src/
│   ├── sentiment_model.py
│   └── pipeline.py
├── notebooks/
├── requirements.txt
└── README.md
```

## Tech Stack
- Python, HuggingFace Transformers (DistilBERT)
- Pandas for data processing
- Power BI for dashboarding
- Apache Kafka (architecture design)

## Results
- 500+ posts processed
- BERT-based sentiment classification (Positive/Negative)
- Subreddit-level breakdown of sentiment trends
- Exported to CSV for Power BI ingestion

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/pipeline.py
```