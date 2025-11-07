# Using Claude Code for Bias Detection Project

## 🤖 What Makes This Project Different

Unlike Project 1 (topic classification), bias detection is:
- **More Subtle:** Bias is in framing, not obvious keywords
- **Context-Dependent:** Same words mean different things
- **Lower Accuracy:** 65-80% is good (not 90%+)
- **More Complex:** Stacked bidirectional LSTMs, class imbalance
- **Ethically Sensitive:** Requires thoughtful deployment

**Use Claude Code to help you navigate this complexity!**

---

## 🎯 Effective Prompting for This Project

### ✅ Good Prompts (Specific, Context-Aware)

```
"Load the AllSides dataset and analyze class imbalance. Show me the distribution and suggest how to handle it."

"Why do we need bidirectional LSTMs for bias detection? Explain with examples of how context matters."

"Build the model with stacked bidirectional LSTMs. Explain why this is more complex than Project 1."

"My Center class is performing poorly (40% F1). Help me debug. Should I use class weights?"

"Analyze these misclassifications. Why did the model predict Right when it's actually Center?"
```

### ❌ Bad Prompts (Missing Context)

```
"Build the model" ❌ (Not specific to bias detection)

"Why is accuracy low?" ❌ (Low compared to what? Context matters!)

"Fix it" ❌ (What specifically needs fixing?)
```

---

## 📋 Phase-by-Phase Prompts

### Phase 1: Data Loading & Exploration (8 minutes)

```bash
# Understanding the dataset
claude-code "Explain the AllSides dataset. What makes it good for bias detection?"

# Loading with awareness of challenges
claude-code "Load the AllSides CSV. Check for: (1) class imbalance, (2) missing values, (3) average article length per bias"

# Exploring patterns
claude-code "Show me sample articles from Left, Center, and Right. What linguistic differences do you notice?"

# Understanding the challenge
claude-code "Why is class imbalance a problem for bias detection? How severe is it in this dataset?"
```

### Phase 2: Preprocessing (7 minutes)

```bash
# Vocabulary size justification
claude-code "Why do we need vocabulary size 20000 for bias (vs 10000 for topics)? Give examples of subtle words that matter."

# Sequence length reasoning
claude-code "Implement tokenization with length 400. Why longer than Project 1? Show what gets cut off at 200 vs 400."

# Stratified splitting
claude-code "Implement stratified train/val/test split. Why is stratification critical for imbalanced data?"

# Verification
claude-code "Verify my preprocessing: Check (1) no data leakage, (2) class distribution maintained, (3) sequences properly padded"
```

### Phase 3: Class Weights (5 minutes)

```bash
# Understanding imbalance
claude-code "Calculate class weights for the imbalanced dataset. Explain what these weights do during training."

# Impact analysis
claude-code "Show me the difference: Training with vs without class weights. What happens to Center class performance?"

# Alternative approaches
claude-code "Besides class weights, what other techniques handle imbalance? (SMOTE, focal loss, etc.) Which is best here?"
```

### Phase 4: Embeddings (5 minutes)

```bash
# Loading large file
claude-code "Load GloVe 300d embeddings. Warn me this takes time and show progress every 100k words."

# Coverage analysis
claude-code "Create embedding matrix. Report: (1) % vocabulary covered, (2) % articles with >90% coverage, (3) most common missing words"

# Justification
claude-code "Why use 300d instead of 100d for bias detection? Give examples where dimensionality matters."
```

### Phase 5: Model Building (10 minutes)

```bash
# Architecture understanding
claude-code "Explain each layer in the stacked bidirectional LSTM. Why is this architecture better for bias than simple LSTM?"

# Building step-by-step
claude-code "Build the model layer by layer, explaining each:
1. Embedding (300d, frozen)
2. Bidirectional LSTM 1 (128, return sequences - why?)
3. Bidirectional LSTM 2 (64 - why smaller?)
4. GlobalMaxPooling (what does this extract?)
5. Dense + Dropout + Output"

# Architecture comparison
claude-code "Compare: (1) Simple LSTM, (2) Bidirectional LSTM, (3) Stacked Bidirectional LSTM. Which is best for bias?"

# Parameter count
claude-code "Print model summary. Explain why this has ~3M parameters vs ~1M in Project 1."
```

### Phase 6: Training (12 minutes)

```bash
# Training setup
claude-code "Set up training with early stopping, LR reduction, and model checkpointing. Apply class weights."

# Monitoring expectations
claude-code "Train the model. What accuracy should I expect? Why is 70% good for bias but poor for topics?"

# Debugging convergence
claude-code "Training is stuck at 40% accuracy. Debug checklist: (1) Data loaded correctly? (2) Labels correct? (3) Class weights applied? (4) Model architecture sound?"

# Overfitting analysis
claude-code "Train accuracy is 85% but validation is 60%. Is this overfitting? How do I fix it without hurting performance?"

# Class-specific issues
claude-code "Left and Right perform well (80% F1) but Center is poor (50% F1). Why? How do I improve Center without hurting others?"
```

### Phase 7: Evaluation (5 minutes)

```bash
# Comprehensive evaluation
claude-code "Evaluate the model. Show: (1) Overall accuracy, (2) Per-class precision/recall/F1, (3) Confusion matrix, (4) Most confused pairs"

# Confusion analysis
claude-code "Generate confusion matrix. Why are Left↔Center and Right↔Center confused more than Left↔Right?"

# Error analysis
claude-code "Show me 5 misclassified articles from each class. Try to identify patterns in errors."

# Comparison to baselines
claude-code "How does this compare to: (1) Random baseline (33%), (2) Majority class baseline, (3) Simple LSTM?"
```

### Phase 8: Real-World Testing (3 minutes)

```bash
# Testing on known sources
claude-code "Test on articles from CNN, Fox News, BBC, Reuters. Does the model's prediction match known source bias?"

# Analyzing disagreements
claude-code "When the model disagrees with known source bias, what linguistic patterns is it detecting? Is it wrong or catching something real?"

# Edge cases
claude-code "Test on: (1) Opinion pieces, (2) Satirical articles, (3) International news, (4) Local news. How does it perform?"
```

---

## 🐛 Debugging with Claude Code

### Issue 1: Low Accuracy (Worse than Random)

```bash
claude-code "Model is stuck at 33% (random guessing). Debug step-by-step:
1. Are labels loaded correctly? Show unique values.
2. Is tokenization working? Show a tokenized example.
3. Are sequences padded correctly? Show shapes.
4. Is the model architecture correct? Show summary.
5. Are class weights applied? Show training logs."
```

### Issue 2: Class Imbalance Problems

```bash
claude-code "My model ignores Center class (0% recall). Diagnose:
1. What's the class distribution?
2. Are class weights calculated correctly?
3. Are they being used in model.fit()?
4. Try increasing Center weight manually, does it help?"
```

### Issue 3: Severe Overfitting

```bash
claude-code "Train acc 90%, val acc 55%. Severe overfitting. Try in order:
1. Increase dropout (try 0.4, 0.5, 0.6)
2. Add L2 regularization
3. Reduce model complexity (fewer LSTM units)
4. Get more training data
Show results for each step."
```

### Issue 4: Slow Training

```bash
claude-code "Training is very slow (30 min per epoch). Optimize:
1. Reduce batch size? (Current: 32)
2. Reduce sequence length? (Current: 400)
3. Use simpler model? (Remove one LSTM layer)
4. Use GPU if available
What's the speed/accuracy tradeoff?"
```

### Issue 5: Out of Memory

```bash
claude-code "Getting OOM errors. Fix by:
1. Reducing batch size to 16
2. Reducing MAX_LEN to 300
3. Using gradient checkpointing
4. Clearing session between runs
Which has least impact on accuracy?"
```

---

## 💡 Advanced Usage for This Project

### Understanding Bias Through the Model

```bash
# Attention visualization (if implemented)
claude-code "Add simple attention. Show which words the model focuses on for Left vs Right predictions."

# Feature importance
claude-code "Extract words most predictive of each bias class. Do they match human intuition about loaded language?"

# Embedding space analysis
claude-code "Visualize word embeddings for politically charged terms. Are 'protesters'/'rioters' close together? Far apart?"
```

### Comparative Analysis

```bash
# Architecture comparison
claude-code "Compare these architectures on bias detection:
1. Simple LSTM
2. Bidirectional LSTM
3. Stacked Bidirectional LSTM
4. LSTM with attention
Show accuracy, training time, and confusion matrices."

# Embedding comparison
claude-code "Compare: (1) GloVe 100d, (2) GloVe 300d, (3) Trainable embeddings. Which gives best bias detection?"

# Preprocessing variations
claude-code "Test impact of: (1) MAX_LEN: 200 vs 300 vs 400, (2) MAX_WORDS: 10k vs 20k vs 30k. Plot accuracy vs parameter."
```

### Ethical Analysis

```bash
# Fairness audit
claude-code "Audit the model for fairness: Does it perform equally well on Left vs Right? Or does it have systematic bias toward one side?"

# Failure analysis
claude-code "Identify systematic failures. Does the model fail on certain topics (economy, immigration, etc.)? Certain sources?"

# Adversarial testing
claude-code "Generate adversarial examples: Can I fool the model by changing a few key words? Which words are most influential?"
```

---

## 🎓 Learning Through Questions

### Ask "Why" Questions About Bias

```bash
"Why is 'illegal aliens' considered right-leaning but 'undocumented immigrants' left-leaning? What does the model learn?"

"Why does context matter? Show how 'protesters' can be neutral or loaded."

"Why is Center hardest to detect? What makes it different from Left and Right?"

"Why do we need longer sequences (400) for bias vs topics (200)?"

"Why is 70% accuracy good for bias but poor for topics?"
```

### Request Explanations with Examples

```bash
"Show me a Left-biased article and explain what linguistic patterns signal Left bias."

"Compare how Left, Center, and Right sources cover the same event. What differs?"

"Give me examples of loaded language that the model should detect."

"Show a misclassified article. Why did the model get it wrong? What confused it?"
```

### Explore Ethical Implications

```bash
"What could go wrong if this model was used for content moderation?"

"How might bad actors try to game this system?"

"Should social media platforms use bias detection? Why or why not?"

"What's the difference between detecting bias and determining truth?"
```

---

## 🚫 What NOT to Do

### Don't Expect Perfect Accuracy
❌ "Why isn't my model getting 90% accuracy?"
✅ "My model is at 72% accuracy. Is this good for bias detection? How do I improve without overfitting?"

### Don't Ignore Class Imbalance
❌ "Just train the model normally"
✅ "The Center class is underrepresented. Help me use class weights effectively."

### Don't Skip Ethical Considerations
❌ "Let's deploy this to auto-filter biased news"
✅ "What are the ethical implications? How should this tool be used responsibly?"

### Don't Compare Directly to Project 1
❌ "Project 1 got 90% but this only gets 70%, something's wrong!"
✅ "Why is bias detection inherently harder than topic classification?"

---

## 🎯 Workshop Timeline with Claude Code

**Minutes 0-8: Data Understanding**
```bash
claude-code "Help me understand the AllSides dataset and the bias detection challenge"
```

**Minutes 8-15: Preprocessing**
```bash
claude-code "Implement preprocessing for bias: larger vocab, longer sequences, stratified split, class weights"
```

**Minutes 15-25: Model Building**
```bash
claude-code "Build stacked bidirectional LSTM. Explain why this architecture suits bias detection."
```

**Minutes 25-37: Training & Debugging**
```bash
claude-code "Train with class weights. Help me interpret results and debug issues."
```

**Minutes 37-45: Evaluation & Ethics**
```bash
claude-code "Evaluate comprehensively. Discuss: Where does it work? Where does it fail? Ethical implications?"
```

---

## 🔥 Pro Tips for This Project

### 1. Manage Expectations
- 70% accuracy is GOOD for bias detection
- Humans only agree ~60% of the time
- Center will perform worst - that's normal

### 2. Understand the Complexity
- Ask Claude Code to explain WHY things are harder
- Request comparisons to Project 1
- Explore what makes bias subtle

### 3. Use Error Analysis
- Don't just look at overall accuracy
- Examine specific failures
- Ask Claude Code to explain misclassifications

### 4. Think About Ethics
- Every design choice has ethical implications
- Ask Claude Code about potential misuses
- Discuss with your team: Should we build this?

### 5. Embrace the Challenge
- This is harder than Project 1 by design
- Struggling is part of learning
- Ask for help when stuck >5 minutes

---

## 📞 Getting Unstuck (Bias-Specific)

### Stuck on Low Accuracy?
```bash
claude-code "My model is at 45% accuracy. Walk me through systematic debugging:
1. Data loading (show samples with labels)
2. Preprocessing (verify tokenization)
3. Model architecture (check layer connections)
4. Training (verify class weights used)
5. Evaluation (check if calculated correctly)"
```

### Confused About Why Bias is Harder?
```bash
claude-code "Explain with concrete examples: Why is bias detection harder than topic classification? Show me articles where bias is subtle."
```

### Unsure About Ethical Implications?
```bash
claude-code "Help me think through the ethics: If I deploy this model, what could go wrong? What safeguards do I need?"
```

---

## ✨ Remember

For bias detection, Claude Code is most helpful when you:
- **Ask about WHY:** Why is this harder? Why these choices?
- **Request comparisons:** How does this differ from Project 1?
- **Seek understanding:** Not just code, but concepts
- **Explore ethics:** What are the implications?
- **Embrace complexity:** Don't expect simple answers

**This is advanced NLP. Claude Code is your guide through the complexity! 🚀**

**Good luck, and remember: Building responsibly is more important than building perfectly! 🌟**
