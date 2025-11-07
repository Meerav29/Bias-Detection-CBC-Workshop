"""
Project 2: Political Bias Detection with LSTM
-------------------------------------------------------
This starter code provides the skeleton for building an LSTM-based bias detector.

Dataset: AllSides News Corpus (3 classes: Left, Center, Right)
Expected Accuracy: 65-80% (harder than topic classification!)
Workshop Duration: 45 minutes

TODO sections mark where you need to implement code.
Use Claude Code to help: "Help me complete the TODO sections for bias detection"

IMPORTANT: This is more challenging than topic classification because bias
           is subtle and context-dependent. Don't expect 90%+ accuracy!
"""

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION - Adjust these parameters as needed
# ============================================================================

# Dataset configuration
MAX_WORDS = 20000        # Larger vocabulary for bias detection (subtle word choices matter!)
MAX_LEN = 400            # Longer sequences for more context
EMBEDDING_DIM = 300      # Use 300d for better semantic understanding

# Model configuration
LSTM_UNITS_1 = 128       # First bidirectional LSTM layer
LSTM_UNITS_2 = 64        # Second bidirectional LSTM layer
DROPOUT_RATE = 0.3       # Dropout rate for LSTM layers
DENSE_DROPOUT = 0.5      # Dropout rate for dense layer
BATCH_SIZE = 32          # Smaller batch size for complex model
EPOCHS = 15              # May need more epochs than topic classification

# File paths
DATA_PATH = 'data/allsides_news_complete.csv'  # Path to AllSides dataset
GLOVE_PATH = 'embeddings/glove.840B.300d.txt'  # Path to GloVe 300d embeddings


# ============================================================================
# PART 1: DATA LOADING AND EXPLORATION
# ============================================================================

def load_data(filepath):
    """
    Load the AllSides bias dataset from CSV file.
    
    Expected columns:
    - 'text' or 'content': Article text
    - 'bias' or 'bias_label': Left (0), Center (1), Right (2)
    
    NOTE: The dataset may be imbalanced (fewer Center articles than Left/Right)
    
    TODO: Load the CSV file and extract text and bias labels
    TODO: Handle missing values
    TODO: Ensure labels are integers (0, 1, 2)
    
    Returns:
        X: List of article texts
        y: numpy array of integer labels (0-2)
    """
    # TODO: Implement data loading
    # Hint: Use pandas to read CSV
    # Hint: Check column names (might vary: 'text' vs 'content', 'bias' vs 'bias_label')
    # Hint: Handle NaN values (drop or fill)
    # Hint: Convert bias labels to integers if needed
    
    pass  # Remove this and add your implementation


def explore_data(X, y):
    """
    Print detailed statistics about the bias dataset.
    
    IMPORTANT: Pay attention to class imbalance!
    
    TODO: Calculate and print:
    - Total number of articles
    - Distribution of bias labels (count and percentage)
    - Average article length per bias category
    - Sample articles from each bias category
    - Identify if dataset is imbalanced
    
    Args:
        X: List of article texts
        y: numpy array of labels
    """
    # TODO: Implement comprehensive data exploration
    # Hint: Use pandas value_counts() for distribution
    # Hint: Calculate percentages to see imbalance
    # Hint: Show sample articles to understand bias patterns
    # Hint: Check average length per class (may reveal patterns)
    
    pass  # Remove this and add your implementation


# ============================================================================
# PART 2: TEXT PREPROCESSING
# ============================================================================

def preprocess_text(X, y, test_size=0.15, val_size=0.15):
    """
    Preprocess text data for bidirectional LSTM input.
    
    IMPORTANT: Use stratified splitting to maintain class balance!
    
    Steps:
    1. Split data with stratification (maintain bias distribution)
    2. Tokenize with larger vocabulary (subtle word choices matter!)
    3. Pad to longer sequences (bias needs more context)
    
    TODO: Use stratified train_test_split
    TODO: Tokenize with larger vocabulary (20,000 words)
    TODO: Pad sequences to longer length (400 tokens)
    
    Args:
        X: List of article texts
        y: numpy array of labels
        test_size: Proportion for test set
        val_size: Proportion for validation set
        
    Returns:
        X_train, X_val, X_test: Padded sequences
        y_train, y_val, y_test: Label arrays
        tokenizer: Fitted tokenizer object
    """
    # TODO: Create and fit tokenizer with MAX_WORDS vocabulary
    # Hint: Bias detection needs larger vocabulary than topic classification
    
    # TODO: Convert texts to sequences
    
    # TODO: Pad sequences to MAX_LEN
    # Hint: Bias needs more context, so longer sequences
    
    # TODO: Stratified split to maintain class balance
    # Hint: Use stratify parameter in train_test_split
    # Hint: Split twice: train/temp, then temp into val/test
    
    pass  # Remove this and add your implementation


def compute_class_weights(y_train):
    """
    Compute class weights to handle imbalanced dataset.
    
    Since Center articles are fewer than Left/Right, we need to
    give them higher weight during training.
    
    TODO: Calculate class weights using sklearn
    TODO: Convert to dictionary format for Keras
    
    Args:
        y_train: Training labels
        
    Returns:
        class_weights: Dictionary mapping class -> weight
    """
    # TODO: Compute class weights
    # Hint: Use sklearn's compute_class_weight
    # Hint: Convert to dict format: {0: weight0, 1: weight1, 2: weight2}
    
    pass  # Remove this and add your implementation


# ============================================================================
# PART 3: EMBEDDING MATRIX
# ============================================================================

def load_glove_embeddings(filepath):
    """
    Load pre-trained GloVe 300d word embeddings.
    
    For bias detection, we use 300d instead of 100d because we need
    better semantic understanding of subtle linguistic differences.
    
    TODO: Read GloVe file and create embeddings dictionary
    TODO: Handle large file size (~2GB for 300d)
    
    Args:
        filepath: Path to GloVe embeddings file
        
    Returns:
        embeddings_index: Dictionary mapping words to embedding vectors
    """
    # TODO: Load GloVe 300d embeddings
    # Hint: File is large (~2GB), may take a minute to load
    # Hint: Each line: word float1 float2 ... float300
    # Hint: Print progress every 100,000 words
    
    pass  # Remove this and add your implementation


def create_embedding_matrix(word_index, embeddings_index):
    """
    Create embedding matrix for our vocabulary using GloVe 300d.
    
    TODO: Create matrix of shape (vocab_size, 300)
    TODO: Fill with GloVe vectors where available
    
    Args:
        word_index: Dictionary from tokenizer
        embeddings_index: Dictionary of GloVe embeddings
        
    Returns:
        embedding_matrix: numpy array of shape (vocab_size, EMBEDDING_DIM)
    """
    # TODO: Initialize embedding matrix
    # Hint: Shape is (len(word_index) + 1, EMBEDDING_DIM)
    
    # TODO: Fill matrix with GloVe vectors
    # Hint: Track how many words are found vs not found in GloVe
    
    pass  # Remove this and add your implementation


# ============================================================================
# PART 4: MODEL BUILDING (STACKED BIDIRECTIONAL LSTM)
# ============================================================================

def build_model(vocab_size, embedding_matrix=None):
    """
    Build stacked bidirectional LSTM for bias detection.
    
    Architecture (MORE COMPLEX than topic classification):
    1. Embedding layer (300d, pre-trained GloVe)
    2. Bidirectional LSTM layer 1 (128 units, return sequences)
    3. Dropout (0.3)
    4. Bidirectional LSTM layer 2 (64 units)
    5. Dropout (0.3)
    6. Global Max Pooling (extract most relevant features)
    7. Dense layer (64 units, ReLU)
    8. Dropout (0.5)
    9. Dense output (3 classes, softmax)
    
    WHY BIDIRECTIONAL: Reading text both forward and backward helps
    capture context that reveals bias (e.g., "protesters" vs "rioters"
    depends on surrounding words)
    
    TODO: Create Sequential model with stacked bidirectional LSTMs
    TODO: Use GlobalMaxPooling1D to extract key features
    TODO: Compile with class_weight support
    
    Args:
        vocab_size: Size of vocabulary
        embedding_matrix: Pre-trained embeddings (optional)
        
    Returns:
        model: Compiled Keras model
    """
    # TODO: Create Sequential model
    model = Sequential()
    
    # TODO: Add Embedding layer
    # Hint: Use embedding_matrix if provided, set trainable=False initially
    # Hint: Input dimensions: vocab_size, output: EMBEDDING_DIM (300)
    
    # TODO: Add first Bidirectional LSTM layer
    # Hint: Use Bidirectional(LSTM(LSTM_UNITS_1, return_sequences=True, dropout=DROPOUT_RATE))
    # Hint: return_sequences=True because we're stacking LSTMs
    
    # TODO: Add second Bidirectional LSTM layer
    # Hint: Use Bidirectional(LSTM(LSTM_UNITS_2, dropout=DROPOUT_RATE))
    # Hint: No return_sequences this time (last LSTM layer)
    
    # TODO: Add GlobalMaxPooling1D (alternative to last LSTM output)
    # NOTE: If using second LSTM without return_sequences, skip pooling
    
    # TODO: Add Dense hidden layer
    # Hint: 64 units, ReLU activation
    
    # TODO: Add Dropout
    # Hint: Use DENSE_DROPOUT (0.5)
    
    # TODO: Add Dense output layer
    # Hint: 3 units (left, center, right)
    # Hint: Use 'softmax' activation
    
    # TODO: Compile model
    # Hint: Loss: 'sparse_categorical_crossentropy'
    # Hint: Optimizer: 'adam'
    # Hint: Metrics: ['accuracy']
    
    return model


# ============================================================================
# PART 5: TRAINING
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val, class_weights=None):
    """
    Train the bidirectional LSTM with class weights and callbacks.
    
    IMPORTANT: Bias detection is HARDER than topic classification!
    - Expect 65-80% accuracy (not 90%+)
    - Training may be slower (bidirectional LSTMs)
    - Watch for overfitting (validation loss increasing)
    
    TODO: Configure callbacks for better training
    TODO: Use class_weights to handle imbalance
    TODO: Monitor both accuracy and per-class performance
    
    Args:
        model: Compiled Keras model
        X_train, y_train: Training data
        X_val, y_val: Validation data
        class_weights: Dictionary of class weights (for imbalanced data)
        
    Returns:
        history: Training history object
    """
    # TODO: Create callbacks
    # Hint: EarlyStopping with patience=5 (may need more than topic classification)
    # Hint: ReduceLROnPlateau with factor=0.5, patience=3
    # Hint: ModelCheckpoint to save best model
    
    # TODO: Fit model
    # Hint: Use BATCH_SIZE (32) and EPOCHS (15)
    # Hint: Pass validation_data=(X_val, y_val)
    # Hint: Pass class_weight if provided (helps with imbalance)
    # Hint: Pass callbacks list
    
    pass  # Remove this and add your implementation


# ============================================================================
# PART 6: EVALUATION AND ANALYSIS
# ============================================================================

def plot_training_history(history):
    """
    Plot training and validation metrics.
    
    WATCH FOR: Overfitting (train accuracy >> val accuracy)
    
    TODO: Create plots for accuracy and loss
    TODO: Add markers for best epoch
    
    Args:
        history: Training history from model.fit()
    """
    # TODO: Create figure with 2 subplots
    # TODO: Plot training and validation accuracy
    # TODO: Plot training and validation loss
    # Hint: Add title, labels, legend, grid
    
    pass  # Remove this and add your implementation


def evaluate_model(model, X_test, y_test):
    """
    Comprehensive evaluation on test set.
    
    IMPORTANT: Look at per-class metrics!
    - Is "Center" harder to detect than "Left" or "Right"?
    - Which biases get confused?
    
    TODO: Generate predictions
    TODO: Calculate overall and per-class metrics
    TODO: Create confusion matrix
    TODO: Analyze which biases are confused
    
    Args:
        model: Trained model
        X_test, y_test: Test data
    """
    # TODO: Get predictions
    # Hint: Use model.predict() then np.argmax()
    
    # TODO: Calculate test accuracy
    
    # TODO: Generate confusion matrix
    # Hint: Show which biases get confused (Left↔Center, Right↔Center, etc.)
    # Hint: Use seaborn heatmap with labels=['Left', 'Center', 'Right']
    
    # TODO: Print classification report
    # Hint: Show precision, recall, F1 for each bias category
    # Hint: Which category performs worst?
    
    pass  # Remove this and add your implementation


def analyze_predictions(model, tokenizer, X_test, y_test, num_samples=10):
    """
    Analyze specific predictions to understand what the model learned.
    
    TODO: Show sample articles with predictions
    TODO: Highlight correct and incorrect predictions
    TODO: Try to identify patterns in errors
    
    Args:
        model: Trained model
        tokenizer: Fitted tokenizer
        X_test, y_test: Test data
        num_samples: Number of samples to analyze
    """
    # TODO: Get predictions for samples
    # TODO: Print articles with true vs predicted labels
    # TODO: Show confidence scores
    # TODO: Identify interesting cases (high confidence errors, edge cases)
    
    pass  # Remove this and add your implementation


def predict_bias(text, model, tokenizer):
    """
    Predict bias of a new article.
    
    TODO: Preprocess text same as training
    TODO: Get prediction with confidence scores
    TODO: Return predicted bias and confidence
    
    Args:
        text: Article text (string)
        model: Trained model
        tokenizer: Fitted tokenizer
        
    Returns:
        predicted_bias: String ('Left', 'Center', or 'Right')
        confidence: Probability of predicted class
        all_probs: Probabilities for all classes
    """
    # TODO: Tokenize and pad text
    
    # TODO: Get prediction
    
    # TODO: Convert to bias label and confidence
    
    pass  # Remove this and add your implementation


# ============================================================================
# PART 7: REAL-WORLD TESTING
# ============================================================================

def test_on_real_articles(model, tokenizer):
    """
    Test the model on real articles from known sources.
    
    This helps us understand:
    - Does the model's predictions match known source bias?
    - What linguistic patterns is it detecting?
    - Where does it fail?
    
    TODO: Test on articles from CNN, Fox News, BBC, Reuters, etc.
    TODO: Compare predictions to known source bias
    TODO: Analyze disagreements
    
    Args:
        model: Trained model
        tokenizer: Fitted tokenizer
    """
    # Sample articles (you can add real ones from different sources)
    test_articles = {
        "CNN (Left-leaning)": "Democrats pushed forward with new legislation...",
        "Fox News (Right-leaning)": "Republicans stood firm against radical proposals...",
        "BBC (Center)": "The government announced new policy measures...",
    }
    
    # TODO: Predict bias for each article
    # TODO: Compare with known source bias
    # TODO: Discuss matches and mismatches
    
    pass  # Remove this and add your implementation


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function for bias detection pipeline.
    
    Steps:
    1. Load and explore AllSides dataset
    2. Preprocess with stratified splitting
    3. Compute class weights (for imbalance)
    4. Load GloVe 300d embeddings
    5. Build stacked bidirectional LSTM
    6. Train with class weights
    7. Evaluate comprehensively
    8. Test on real articles
    9. Discuss ethical implications
    
    TODO: Complete each step by calling the functions above
    """
    
    print("=" * 70)
    print("POLITICAL BIAS DETECTION WITH LSTM")
    print("=" * 70)
    print("\nNOTE: This is more challenging than topic classification!")
    print("Expected accuracy: 65-80% (bias is subtle and context-dependent)\n")
    
    # Step 1: Load data
    print("\n[1/8] Loading AllSides bias dataset...")
    # TODO: Call load_data() and explore_data()
    # NOTE: Pay attention to class imbalance!
    
    # Step 2: Preprocess
    print("\n[2/8] Preprocessing text (larger vocab, longer sequences)...")
    # TODO: Call preprocess_text()
    
    # Step 3: Compute class weights
    print("\n[3/8] Computing class weights (handling imbalance)...")
    # TODO: Call compute_class_weights()
    
    # Step 4: Load embeddings
    print("\n[4/8] Loading GloVe 300d embeddings (this may take a minute)...")
    # TODO: Call load_glove_embeddings() and create_embedding_matrix()
    
    # Step 5: Build model
    print("\n[5/8] Building stacked bidirectional LSTM...")
    # TODO: Call build_model()
    # TODO: Print model summary (should be ~2-3M parameters)
    
    # Step 6: Train
    print("\n[6/8] Training model (this will take longer than topic classification)...")
    # TODO: Call train_model() with class_weights
    
    # Step 7: Visualize
    print("\n[7/8] Visualizing training history...")
    # TODO: Call plot_training_history()
    
    # Step 8: Evaluate
    print("\n[8/8] Evaluating on test set...")
    # TODO: Call evaluate_model()
    # TODO: Call analyze_predictions()
    # TODO: Call test_on_real_articles()
    
    # Final discussion
    print("\n" + "=" * 70)
    print("ETHICAL CONSIDERATIONS")
    print("=" * 70)
    print("""
This model detects patterns, not truth. Important reminders:

1. Bias detection ≠ fact-checking
2. What is "center" is culturally defined
3. Predictions should inform, not dictate decisions
4. Technology alone cannot solve media bias
5. Always maintain human oversight

Questions to discuss:
- How should this technology be used responsibly?
- What could go wrong if misused?
- Who should control bias detection tools?
    """)
    
    print("\nTraining complete! Remember: Use this tool responsibly and ethically.")


if __name__ == "__main__":
    # Run the bias detection pipeline
    # Use Claude Code for help at each step:
    # "Help me load and explore the AllSides dataset"
    # "Build the bidirectional LSTM architecture"
    # "Why is my Center class performing poorly?"
    # "Explain the ethical implications of bias detection"
    
    main()
