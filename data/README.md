# Data Directory

Place your AllSides bias dataset here as `allsides_news_complete.csv`

## Download Options:

### Option 1: Qbias Repository (Recommended)
```bash
git clone https://github.com/irgroup/Qbias.git
cp Qbias/data/allsides_news_complete.csv .
```

Or download directly:
```bash
wget https://raw.githubusercontent.com/irgroup/Qbias/main/data/allsides_news_complete.csv
```

### Option 2: Article Bias Prediction Dataset
https://github.com/ramybaly/Article-Bias-Prediction

### Option 3: Run the Scraper (Get Latest Data)
Use the AllsidesDataCrawl.ipynb from Qbias repo to scrape current data

## Expected Format:
- Column: `text` or `content` (article text)
- Column: `bias` or `bias_label` (Left=0, Center=1, Right=2)
- Column: `headline` (optional)
- Column: `source` (optional)
- Column: `date` (optional)

## Size:
- Approximately 21,747 articles total
- Left: ~10,273 articles
- Center: ~4,252 articles (NOTE: Imbalanced!)
- Right: ~7,222 articles

## Important Notes:
1. **Class Imbalance:** Center class is underrepresented
2. **Use Class Weights:** Essential for handling imbalance
3. **Stratified Splitting:** Maintain class distribution in train/val/test
