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
    # Load CSV file
    df = pd.read_csv(filepath)

    # Check for column names (may vary across datasets)
    text_col = 'text' if 'text' in df.columns else 'content'
    bias_col = 'bias_rating' if 'bias_rating' in df.columns else ('bias' if 'bias' in df.columns else 'bias_label')

    # Drop rows with missing values
    df = df[[text_col, bias_col]].dropna()

    # Map bias labels to integers
    bias_mapping = {'left': 0, 'center': 1, 'right': 2}

    # Convert bias labels to lowercase and map
    if df[bias_col].dtype == 'object':
        df[bias_col] = df[bias_col].str.lower().map(bias_mapping)

    # Extract features and labels
    X = df[text_col].tolist()
    y = np.array(df[bias_col], dtype=int)

    print(f"Loaded {len(X)} articles")

    return X, y


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
    label_names = {0: 'Left', 1: 'Center', 2: 'Right'}

    print(f"\n{'='*70}")
    print("DATASET STATISTICS")
    print(f"{'='*70}")
    print(f"Total articles: {len(X)}")

    # Calculate distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nClass Distribution:")
    for label, count in zip(unique, counts):
        percentage = (count / len(y)) * 100
        print(f"  {label_names[label]:>6}: {count:>6} articles ({percentage:>5.2f}%)")

    # Check for imbalance
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count
    print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")
    if imbalance_ratio > 1.5:
        print("  ⚠️  Dataset is imbalanced - class weights recommended!")

    # Average article length per class
    print(f"\nAverage Article Length (words):")
    for label in unique:
        mask = y == label
        avg_len = np.mean([len(text.split()) for i, text in enumerate(X) if mask[i]])
        print(f"  {label_names[label]:>6}: {avg_len:>6.1f} words")

    # Sample articles from each class
    print(f"\nSample Articles (first 200 characters):")
    for label in unique:
        indices = np.where(y == label)[0]
        sample_idx = indices[0]
        sample_text = X[sample_idx][:200] + "..."
        print(f"\n  {label_names[label]}:")
        print(f"    {sample_text}")


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
    # Create and fit tokenizer with larger vocabulary
    print(f"Creating tokenizer with {MAX_WORDS} words vocabulary...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(X)

    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(X)

    # Pad sequences to fixed length
    X_padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    print(f"Padded sequences to max length: {MAX_LEN}")
    print(f"Vocabulary size: {min(len(tokenizer.word_index) + 1, MAX_WORDS)}")

    # Stratified split: train/temp (70/30)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_padded, y, test_size=(test_size + val_size), stratify=y, random_state=42
    )

    # Stratified split: temp into val/test (15/15)
    val_ratio = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_ratio), stratify=y_temp, random_state=42
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)} ({len(X_train)/len(X_padded)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} ({len(X_val)/len(X_padded)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} ({len(X_test)/len(X_padded)*100:.1f}%)")

    # Verify stratification maintained class balance
    print(f"\nClass distribution maintained:")
    for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        unique, counts = np.unique(y_split, return_counts=True)
        percentages = [f"{c/len(y_split)*100:.1f}%" for c in counts]
        print(f"  {split_name}: {' / '.join(percentages)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, tokenizer


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
    # Get unique classes
    classes = np.unique(y_train)

    # Compute balanced class weights
    weights = compute_class_weight('balanced', classes=classes, y=y_train)

    # Convert to dictionary format for Keras
    class_weights = {i: weights[i] for i in range(len(classes))}

    label_names = {0: 'Left', 1: 'Center', 2: 'Right'}
    print(f"\nClass weights (to handle imbalance):")
    for cls, weight in class_weights.items():
        print(f"  {label_names[cls]:>6}: {weight:.3f}")

    return class_weights


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
    print(f"Loading GloVe embeddings from {filepath}...")
    print("This may take a minute (2.2M word vectors)...")

    embeddings_index = {}
    errors = 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                # Split line into word and coefficients
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')

                # Verify correct dimensions (300d)
                if len(coefs) == EMBEDDING_DIM:
                    embeddings_index[word] = coefs
                else:
                    errors += 1

            except (ValueError, IndexError):
                # Handle malformed lines
                errors += 1
                continue

            # Print progress every 100,000 words
            if (i + 1) % 100000 == 0:
                print(f"  Loaded {i + 1:,} word vectors...")

    print(f"\nLoaded {len(embeddings_index):,} word vectors (300d)")
    if errors > 0:
        print(f"Skipped {errors} malformed lines")

    return embeddings_index


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
    # Calculate vocabulary size (limit to MAX_WORDS)
    vocab_size = min(len(word_index) + 1, MAX_WORDS + 1)

    # Initialize embedding matrix with zeros
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))

    # Fill matrix with GloVe vectors
    found = 0
    not_found = 0

    for word, i in word_index.items():
        if i >= MAX_WORDS:
            continue

        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Found word in GloVe
            embedding_matrix[i] = embedding_vector
            found += 1
        else:
            # Word not in GloVe - keep zero vector
            not_found += 1

    # Print coverage statistics
    total = found + not_found
    coverage = (found / total) * 100 if total > 0 else 0

    print(f"\nEmbedding matrix created:")
    print(f"  Shape: {embedding_matrix.shape}")
    print(f"  Words found in GloVe: {found:,} ({coverage:.1f}%)")
    print(f"  Words not found: {not_found:,} ({100-coverage:.1f}%)")

    return embedding_matrix


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
    model = Sequential()

    # Embedding layer with pre-trained GloVe
    if embedding_matrix is not None:
        model.add(Embedding(
            vocab_size,
            EMBEDDING_DIM,
            weights=[embedding_matrix],
            trainable=False  # Freeze pre-trained embeddings
        ))
    else:
        model.add(Embedding(vocab_size, EMBEDDING_DIM))

    # First Bidirectional LSTM layer (return sequences for stacking)
    model.add(Bidirectional(LSTM(
        LSTM_UNITS_1,
        return_sequences=True,
        dropout=DROPOUT_RATE,
        recurrent_dropout=DROPOUT_RATE
    )))

    # Second Bidirectional LSTM layer
    model.add(Bidirectional(LSTM(
        LSTM_UNITS_2,
        dropout=DROPOUT_RATE,
        recurrent_dropout=DROPOUT_RATE
    )))

    # Dense hidden layer
    model.add(Dense(64, activation='relu'))

    # Dropout for regularization
    model.add(Dropout(DENSE_DROPOUT))

    # Output layer (3 classes: Left, Center, Right)
    model.add(Dense(3, activation='softmax'))

    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

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
    # Configure callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'best_bias_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    print(f"\nTraining model with:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Max epochs: {EPOCHS}")
    print(f"  Class weights: {'Yes' if class_weights else 'No'}")
    print(f"  Early stopping patience: 5")

    # Train model
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    return history


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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Find best epoch (lowest validation loss)
    best_epoch = np.argmin(history.history['val_loss'])

    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    ax1.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label='Best Epoch')
    ax1.plot(best_epoch, history.history['val_accuracy'][best_epoch], 'r*', markersize=15)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    ax2.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.7, label='Best Epoch')
    ax2.plot(best_epoch, history.history['val_loss'][best_epoch], 'r*', markersize=15)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    print(f"\nTraining history plot saved: training_history.png")
    print(f"Best epoch: {best_epoch + 1} (val_loss: {history.history['val_loss'][best_epoch]:.4f})")
    plt.show()


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
    # Get predictions
    print("\nEvaluating model on test set...")
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate test accuracy
    test_accuracy = np.mean(y_pred == y_test)
    print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

    # Classification report
    label_names = ['Left', 'Center', 'Right']
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix - Bias Detection', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\nConfusion matrix saved: confusion_matrix.png")
    plt.show()

    # Analyze confusion patterns
    print("\nConfusion Analysis:")
    for i, true_label in enumerate(label_names):
        for j, pred_label in enumerate(label_names):
            if i != j and cm[i, j] > 0:
                confusion_rate = cm[i, j] / cm[i].sum() * 100
                print(f"  {true_label} → {pred_label}: {cm[i, j]} ({confusion_rate:.1f}%)")

    return y_pred, y_pred_probs


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
    label_names = {0: 'Left', 1: 'Center', 2: 'Right'}

    # Get predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate confidence scores
    confidences = np.max(y_pred_probs, axis=1)
    correct_mask = y_pred == y_test

    # Show correct predictions
    print("\n" + "="*70)
    print("SAMPLE CORRECT PREDICTIONS")
    print("="*70)

    correct_indices = np.where(correct_mask)[0][:num_samples//2]
    for idx in correct_indices:
        confidence = confidences[idx]
        true_label = label_names[y_test[idx]]
        pred_label = label_names[y_pred[idx]]
        all_probs = y_pred_probs[idx]

        print(f"\nTrue: {true_label} | Predicted: {pred_label} | Confidence: {confidence:.2%}")
        print(f"Probabilities - Left: {all_probs[0]:.2%}, Center: {all_probs[1]:.2%}, Right: {all_probs[2]:.2%}")

    # Show incorrect predictions
    print("\n" + "="*70)
    print("SAMPLE INCORRECT PREDICTIONS")
    print("="*70)

    incorrect_indices = np.where(~correct_mask)[0][:num_samples//2]
    for idx in incorrect_indices:
        confidence = confidences[idx]
        true_label = label_names[y_test[idx]]
        pred_label = label_names[y_pred[idx]]
        all_probs = y_pred_probs[idx]

        print(f"\nTrue: {true_label} | Predicted: {pred_label} | Confidence: {confidence:.2%}")
        print(f"Probabilities - Left: {all_probs[0]:.2%}, Center: {all_probs[1]:.2%}, Right: {all_probs[2]:.2%}")

    # High-confidence errors
    high_conf_errors = np.where((~correct_mask) & (confidences > 0.7))[0]
    print(f"\n" + "="*70)
    print(f"HIGH-CONFIDENCE ERRORS: {len(high_conf_errors)} cases (>70% confidence but wrong)")
    print("="*70)

    if len(high_conf_errors) > 0:
        for idx in high_conf_errors[:3]:
            confidence = confidences[idx]
            true_label = label_names[y_test[idx]]
            pred_label = label_names[y_pred[idx]]
            print(f"  True: {true_label} → Predicted: {pred_label} (Confidence: {confidence:.2%})")

    # Confidence analysis
    avg_conf_correct = np.mean(confidences[correct_mask])
    avg_conf_incorrect = np.mean(confidences[~correct_mask])

    print(f"\nConfidence Analysis:")
    print(f"  Average confidence on correct predictions: {avg_conf_correct:.2%}")
    print(f"  Average confidence on incorrect predictions: {avg_conf_incorrect:.2%}")


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
    label_names = {0: 'Left', 1: 'Center', 2: 'Right'}

    # Tokenize and pad text
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')

    # Get prediction
    probs = model.predict(padded, verbose=0)[0]
    predicted_class = np.argmax(probs)

    # Convert to label and confidence
    predicted_bias = label_names[predicted_class]
    confidence = probs[predicted_class]

    return predicted_bias, confidence, probs


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
    # Sample articles from various sources with known bias
    test_articles = {
        "CNN (Left-leaning)": "Progressive Democrats unveiled an ambitious climate plan today, advocating for sweeping reforms to address the climate crisis. Environmental activists praised the comprehensive approach to tackling carbon emissions and promoting renewable energy.",

        "Fox News (Right-leaning)": "Conservative Republicans defended traditional values and criticized the radical left's agenda to fundamentally transform America. Patriots rallied to protect freedom and preserve constitutional rights against government overreach.",

        "BBC (Center)": "The government announced new policy measures today following extensive consultations with stakeholders. Officials indicated that the reforms aim to balance economic growth with social welfare considerations.",

        "Reuters (Center)": "Markets responded to the Federal Reserve's latest interest rate decision, with analysts noting mixed signals in the economic data. Both business leaders and consumer advocates are watching the situation closely.",

        "MSNBC (Left-leaning)": "Advocates for social justice marched in support of progressive policies, demanding action on healthcare reform and income inequality. Community organizers emphasized the need for systemic change to address structural barriers.",

        "Breitbart (Right-leaning)": "Patriots stood firm against the liberal establishment's attempts to undermine American sovereignty. Conservatives mobilized to defend borders and resist the globalist agenda threatening national security.",
    }

    print("\n" + "="*70)
    print("TESTING ON REAL-WORLD ARTICLES")
    print("="*70)

    matches = 0
    total = 0
    expected_bias = {
        "CNN (Left-leaning)": "Left",
        "Fox News (Right-leaning)": "Right",
        "BBC (Center)": "Center",
        "Reuters (Center)": "Center",
        "MSNBC (Left-leaning)": "Left",
        "Breitbart (Right-leaning)": "Right",
    }

    for source, article in test_articles.items():
        predicted_bias, confidence, all_probs = predict_bias(article, model, tokenizer)
        expected = expected_bias[source]

        match = "✓" if predicted_bias == expected else "✗"
        if predicted_bias == expected:
            matches += 1
        total += 1

        print(f"\n{source}")
        print(f"  Expected: {expected}")
        print(f"  Predicted: {predicted_bias} (Confidence: {confidence:.2%}) {match}")
        print(f"  Probabilities - Left: {all_probs[0]:.2%}, Center: {all_probs[1]:.2%}, Right: {all_probs[2]:.2%}")
        print(f"  Article: {article[:100]}...")

    accuracy = matches / total * 100
    print(f"\n" + "="*70)
    print(f"Source Bias Matching: {matches}/{total} ({accuracy:.1f}%)")
    print("="*70)
    print("\nNote: Predictions should match known source bias, but remember:")
    print("  - Individual articles may not reflect overall source bias")
    print("  - Model learned from training data, not source reputation")
    print("  - Bias detection is subtle and context-dependent")


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
    X, y = load_data(DATA_PATH)
    explore_data(X, y)

    # Step 2: Preprocess
    print("\n[2/8] Preprocessing text (larger vocab, longer sequences)...")
    X_train, X_val, X_test, y_train, y_val, y_test, tokenizer = preprocess_text(X, y)

    # Step 3: Compute class weights
    print("\n[3/8] Computing class weights (handling imbalance)...")
    class_weights = compute_class_weights(y_train)

    # Step 4: Load embeddings
    print("\n[4/8] Loading GloVe 300d embeddings (this may take a minute)...")
    embeddings_index = load_glove_embeddings(GLOVE_PATH)
    embedding_matrix = create_embedding_matrix(tokenizer.word_index, embeddings_index)

    # Step 5: Build model
    print("\n[5/8] Building stacked bidirectional LSTM...")
    vocab_size = min(len(tokenizer.word_index) + 1, MAX_WORDS + 1)
    model = build_model(vocab_size, embedding_matrix)
    print("\nModel Architecture:")
    model.summary()

    # Step 6: Train
    print("\n[6/8] Training model (this will take longer than topic classification)...")
    history = train_model(model, X_train, y_train, X_val, y_val, class_weights)

    # Step 7: Visualize
    print("\n[7/8] Visualizing training history...")
    plot_training_history(history)

    # Step 8: Evaluate
    print("\n[8/8] Evaluating on test set...")
    y_pred, y_pred_probs = evaluate_model(model, X_test, y_test)
    analyze_predictions(model, tokenizer, X_test, y_test, num_samples=10)
    test_on_real_articles(model, tokenizer)

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
