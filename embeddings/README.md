# Project 2: Political Bias Detection with LSTM

## 🎯 Objective
Build a sophisticated deep learning system that detects political bias (Left, Center, Right) in news articles using stacked bidirectional LSTM neural networks.

**⚠️ IMPORTANT:** This is significantly harder than topic classification. Bias is subtle, context-dependent, and subjective. Expect 65-80% accuracy, not 90%+!

## 📦 What's Included

```
project2_bias_detection/
├── starter_code.py          # Main implementation with TODO sections
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── CLAUDE_CODE_GUIDE.md     # How to use Claude Code effectively
├── ETHICS.md                # Important ethical considerations
├── data/                    # Place AllSides dataset here
└── embeddings/              # Place GloVe 300d embeddings here
```

## 🚀 Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Download Dataset

**Primary Source: AllSides via GitHub**

```bash
# Clone the Qbias repository
git clone https://github.com/irgroup/Qbias.git

# OR download directly
wget https://raw.githubusercontent.com/irgroup/Qbias/main/data/allsides_news_complete.csv

# Place in data/ folder
mv allsides_news_complete.csv data/
```

**Alternative Source:**
```bash
git clone https://github.com/ramybaly/Article-Bias-Prediction.git
# Use the JSON files in their data/ folder
```

### Step 3: Download GloVe 300d Embeddings

**Download from Stanford NLP:**
```bash
# Download (warning: ~2GB file)
wget http://nlp.stanford.edu/data/glove.840B.300d.zip

# Extract
unzip glove.840B.300d.zip

# Move to embeddings folder
mv glove.840B.300d.txt embeddings/
```

**Note:** This step is strongly recommended (not optional like Project 1) because bias detection needs high-quality semantic understanding.

### Step 4: Start Coding with Claude Code

```bash
claude-code "Help me build the bias detection system"
```

## 📝 Implementation Guide

The `starter_code.py` file has **8 main parts** with TODO sections:

### Part 1: Data Loading (TODO)
- Load AllSides CSV
- Handle potential imbalance (fewer Center articles)
- Explore bias distribution

**Claude Code Prompt:**
```
"Load the AllSides dataset and show me the distribution of bias labels. Is it imbalanced?"
```

### Part 2: Text Preprocessing (TODO)
- Tokenize with larger vocabulary (20,000 words)
- Pad to longer sequences (400 tokens)
- Use stratified splitting to maintain class balance

**Claude Code Prompt:**
```
"Implement preprocessing for bias detection with vocab=20000 and length=400. Use stratified splitting."
```

### Part 3: Class Weights (TODO)
- Compute weights to handle class imbalance
- Give higher weight to underrepresented classes

**Claude Code Prompt:**
```
"Calculate class weights to handle imbalanced dataset"
```

### Part 4: Embedding Matrix (TODO)
- Load GloVe 300d embeddings (~2GB, be patient!)
- Create embedding matrix for vocabulary

**Claude Code Prompt:**
```
"Load GloVe 300d embeddings and create embedding matrix. Show coverage statistics."
```

### Part 5: Model Building (TODO)
- Build stacked bidirectional LSTM
- Use GlobalMaxPooling to extract features
- More complex than topic classification!

**Claude Code Prompt:**
```
"Build stacked bidirectional LSTM with this architecture:
- Embedding (300d from GloVe)
- Bidirectional LSTM (128 units, return sequences, dropout 0.3)
- Bidirectional LSTM (64 units, dropout 0.3)
- Dense (64 units, ReLU, dropout 0.5)
- Output (3 classes, softmax)"
```

### Part 6: Training (TODO)
- Train with class weights
- Use callbacks for early stopping
- Monitor for overfitting

**Claude Code Prompt:**
```
"Train the model with class weights and early stopping. Why is training slower than Project 1?"
```

### Part 7: Evaluation (TODO)
- Generate confusion matrix
- Calculate per-class metrics
- Analyze which biases get confused

**Claude Code Prompt:**
```
"Evaluate model and show confusion matrix. Which bias class performs worst and why?"
```

### Part 8: Real-World Testing (TODO)
- Test on articles from CNN, Fox News, BBC, Reuters
- Compare predictions to known source bias
- Discuss ethical implications

**Claude Code Prompt:**
```
"Test on real articles from different sources. Does it match known source bias?"
```

## 🎓 Learning Objectives

By completing this project, you'll learn:
- ✅ How bias manifests in language
- ✅ Bidirectional LSTM architecture
- ✅ Handling imbalanced datasets
- ✅ Context-dependent classification
- ✅ Ethical implications of bias detection
- ✅ Limitations of automated content analysis

## 🔧 Configuration

Key parameters in `starter_code.py`:

```python
MAX_WORDS = 20000        # Larger vocab (subtle words matter!)
MAX_LEN = 400            # Longer sequences (need more context)
EMBEDDING_DIM = 300      # Use 300d for better semantics
LSTM_UNITS_1 = 128       # First BiLSTM layer
LSTM_UNITS_2 = 64        # Second BiLSTM layer
BATCH_SIZE = 32          # Smaller for complex model
EPOCHS = 15              # May need more than Project 1
```

## 🎯 Success Criteria

Your model should achieve:
- ✅ **Test accuracy:** 65-80% (this is good for bias detection!)
- ✅ **Reasonable per-class metrics:** All classes >60% F1-score
- ✅ **Understanding of confusion:** Know why Center is hardest
- ✅ **Ethical awareness:** Understand limitations and risks

## 🐛 Common Issues & Solutions

### Issue 1: Dataset not found
**Solution:** Make sure `allsides_news_complete.csv` is in `data/` folder

### Issue 2: Class imbalance causing poor Center performance
**Solution:** Use class weights (already in code)

### Issue 3: Model stuck at 33% accuracy (random guessing)
**Solution:**
- Check if data loaded correctly
- Verify preprocessing (tokenization, padding)
- Ensure class weights are applied
- Try training longer

### Issue 4: Overfitting (train acc >> val acc)
**Solution:**
- Increase dropout rates
- Add more regularization
- Use data augmentation
- Get more training data

### Issue 5: Out of memory
**Solution:**
- Reduce BATCH_SIZE to 16
- Reduce MAX_LEN to 300
- Close other applications

### Issue 6: GloVe file takes forever to load
**Solution:**
- This is normal (~2GB file)
- Should take 1-2 minutes
- Consider caching the embedding matrix

## ⚠️ Critical Ethical Considerations

**READ ETHICS.md BEFORE DEPLOYING THIS MODEL!**

Key points:
1. **This detects patterns, not truth**
2. **Bias detection ≠ fact-checking**
3. **"Center" is culturally defined**
4. **Potential for misuse and censorship**
5. **Always maintain human oversight**

## 💡 Why This Is Harder Than Project 1

### Subtle Signals
- Topic: "basketball" = obviously sports
- Bias: "illegal aliens" vs "undocumented immigrants" = same concept, different frame

### Context Dependency
- "Protesters" can be neutral or loaded depending on context
- Same words mean different things in different articles

### Lower Expected Accuracy
- Topic classification: 85-92% is normal
- Bias detection: 65-80% is good!
- 70% accuracy means you're doing well

### Longer Training Time
- Bidirectional LSTMs are 2x slower
- Longer sequences take more memory
- More epochs needed for convergence

## 🚀 Extensions (After Completing Core)

### Easy (1-2 hours)
- Add attention mechanism
- Visualize important words
- Build Streamlit interface
- Compare left vs right coverage of same event

### Intermediate (3-8 hours)
- Multi-task learning (topic + bias)
- Fine-grained bias (framing, omission, etc.)
- Temporal bias tracking
- Cross-source event analysis

### Advanced (2+ weeks)
- Use BERT/RoBERTa transformers
- Explainability with LIME/SHAP
- Cross-lingual bias detection
- Build bias neutralization system

## 📚 Resources

- **AllSides:** https://www.allsides.com/media-bias
- **Research Paper:** "We Can Detect Your Bias" (Baly et al., EMNLP 2020)
- **BiasLab Dataset:** https://github.com/ksolaiman/PoliticalBiasCorpus
- **GloVe Embeddings:** https://nlp.stanford.edu/projects/glove/

## 🆘 Need Help?

### 1. Use Claude Code
```bash
claude-code "I'm stuck on [specific issue]. Here's my code: [paste]"
```

### 2. Check CLAUDE_CODE_GUIDE.md
Specific prompts for each phase of the project

### 3. Read ETHICS.md
Understand the broader implications of what you're building

### 4. Review Starter Code Comments
Detailed hints and explanations throughout

## 🎯 Discussion Questions

After completing the project, discuss:
1. Why is "Center" harder to detect than "Left" or "Right"?
2. What linguistic patterns did the model learn?
3. How does context change the meaning of words?
4. What are the risks of deploying this technology?
5. Who should control bias detection tools?
6. Can technology solve media bias, or just help us understand it?

## 🎉 Ready to Start?

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets (AllSides + GloVe 300d)

# 3. Start coding
claude-code "Let's build a bias detector! Help me understand the dataset first."
```

**Remember: This is challenging, expect lower accuracy than Project 1. Focus on learning, not perfection! 🚀**

**⚠️ Important: Read ETHICS.md before deploying this model in any real application!**
