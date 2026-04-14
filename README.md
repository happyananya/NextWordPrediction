# NextWordPrediction

A deep learning project that predicts the next word in a sequence using LSTM neural networks trained on classical literature. This project demonstrates practical NLP techniques including text preprocessing, tokenization, and sequence modeling.

## 📋 Table of Contents

- [Overview](#overview)
- [Project Motivation](#project-motivation)
- [Technical Architecture](#technical-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Technologies Used](#technologies-used)

## Overview

NextWordPrediction is an LSTM-based text prediction model that learns linguistic patterns from classic literature (Sherlock Holmes stories) and generates contextually relevant next words. The model can predict multiple words sequentially, enabling creative text generation applications.

**Key Features:**
- LSTM-based sequence learning architecture
- Word-level tokenization and embedding
- Multi-word prediction capability
- Pre-trained on Sherlock Holmes corpus
- Simple, extensible Python implementation

## Project Motivation

Language modeling is fundamental to modern NLP applications including:
- Autocomplete systems
- Text generation
- Machine translation
- Speech recognition

This project explores how LSTM networks can capture sequential dependencies in natural language by learning from patterns in existing text. Using classic literature provides a rich, well-structured dataset with diverse vocabulary and narrative patterns.

## Technical Architecture

### Model Components

1. **Embedding Layer** (100 dimensions)
   - Converts tokenized word indices to dense vector representations
   - Captures semantic relationships between words
   - Input length: max_sequence_length - 1

2. **LSTM Layer** (150 units)
   - Processes sequential input to learn temporal dependencies
   - Maintains long-term context for word prediction
   - Bidirectional learning of linguistic patterns

3. **Output Dense Layer**
   - Fully connected layer with softmax activation
   - Output size: total vocabulary size
   - Produces probability distribution over all possible next words

### Architecture Diagram

```
Input Sequences (padded)
        ↓
Embedding Layer (100 dimensions)
        ↓
LSTM Layer (150 units)
        ↓
Dense Layer (vocab_size, softmax)
        ↓
Predicted Word Index
```

## Dataset

- **Source:** Sherlock Holmes stories (plain text)
- **File:** `sherlock-holm.es_stories_plain-text_advs.txt`
- **Processing:**
  - Records split by newbreaks
  - Each line tokenized into sequences
  - N-gram sequences generated from every position
  - Sequences padded to uniform length

**Data Statistics:**
- Total unique words in vocabulary: variable (depends on tokenization)
- Max sequence length: dynamically calculated
- Training samples: N-gram sequences derived from entire corpus

## Installation

### Prerequisites

- Python 3.7+
- pip or conda

### Dependencies

```bash
pip install numpy tensorflow keras
```

### Clone Repository

```bash
git clone https://github.com/happyananya/NextWordPrediction.git
cd NextWordPrediction
```

## Usage

### Running the Complete Pipeline

Execute the Jupyter notebook to run the entire workflow:

```bash
jupyter notebook PredictionModel.ipynb
```

### Step-by-Step Execution

1. **Data Loading**
   ```python
   with open('sherlock-holm.es_stories_plain-text_advs.txt', 'r', encoding='utf-8') as file:
       text = file.read()
   ```

2. **Tokenization**
   ```python
   from tensorflow.keras.preprocessing.text import Tokenizer
   tokenizer = Tokenizer()
   tokenizer.fit_on_texts([text])
   ```

3. **Sequence Generation**
   - Creates n-gram sequences of varying lengths
   - Each sequence: (word_1, word_2, ..., word_n) → predict word_(n+1)

4. **Model Training**
   ```python
   model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
   model.fit(X, y, epochs=100, verbose=1)
   ```

5. **Text Generation**
   ```python
   seed_text = "I will leave if they"
   next_words = 3
   # Model predicts the next 3 words
   ```

### Making Predictions

```python
# Use trained model with seed text
seed_text = "Your seed phrase here"
next_words = 5  # Number of words to predict

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=-1)
    
    # Find predicted word
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            seed_text += " " + word
            break

print(seed_text)
```

## Project Structure

```
NextWordPrediction/
├── PredictionModel.ipynb           # Main model training and prediction notebook
├── sherlock-holm.es_stories_plain-text_advs.txt  # Training dataset
├── README.md                       # Project documentation
└── .gitignore                      # Git ignore rules
```

## Model Details

### Training Configuration

- **Loss Function:** Categorical Crossentropy
  - Measures difference between predicted probability distribution and ground truth
  - Suitable for multi-class classification (predicting one word from vocabulary)

- **Optimizer:** Adam
  - Adaptive learning rate optimization
  - Efficiently handles sparse gradients

- **Metrics:** Accuracy
  - Measures percentage of correctly predicted words

- **Epochs:** 100
  - Number of complete passes through training data
  - Can be adjusted based on validation performance

### Data Preprocessing

1. **Tokenization**
   - Converts words to integer indices
   - Builds vocabulary mapping

2. **Sequence Padding**
   - Pads sequences to uniform length using pre-padding
   - Maintains temporal relationships

3. **Sequence Splitting**
   - Separates features (X): all words except last
   - Separates targets (y): last word in sequence

4. **One-Hot Encoding**
   - Converts target indices to categorical vectors
   - Enables probability-based output calculation

## Results

The model learns to:
- Predict contextually appropriate next words
- Capture character names and common phrases from Sherlock Holmes
- Generate multi-word sequences with reasonable grammatical structure
- Improve predictions with longer context windows

**Performance Metrics:**
- Accuracy improves with training epochs
- Model demonstrates understanding of narrative patterns
- Can generate plausible continuations from seed text

## Future Enhancements

### Model Improvements
- [ ] Bidirectional LSTM for better context understanding
- [ ] Attention mechanisms to focus on relevant context
- [ ] Multiple LSTM layers for deeper feature learning
- [ ] Dropout layers to prevent overfitting
- [ ] Batch normalization for training stability

### Data Enhancements
- [ ] Multiple literature sources for diverse vocabulary
- [ ] Character-level tokenization for handling rare words
- [ ] Data augmentation techniques
- [ ] Preprocessing: stemming, lemmatization, POS tagging

### Functionality Expansions
- [ ] Beam search for better word selection
- [ ] Temperature-based sampling for diverse outputs
- [ ] Interactive web interface for predictions
- [ ] Model export for deployment
- [ ] Performance optimization with TF Lite

### Evaluation
- [ ] BLEU score for text quality
- [ ] Perplexity measurement
- [ ] User studies for output quality
- [ ] Comparison with transformer-based models

## Technologies Used

| Technology | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.7+ | Core programming language |
| **TensorFlow** | 2.x | Deep learning framework |
| **Keras** | 2.x | High-level neural network API |
| **NumPy** | Latest | Numerical computations |
| **Jupyter Notebook** | Latest | Interactive development environment |

## Key Concepts

### LSTM (Long Short-Term Memory)
- Specialized RNN architecture for sequence learning
- Maintains long-term dependencies through cell state
- Solves vanishing gradient problem in vanilla RNNs

### Word Embeddings
- Dense vector representations of words
- Capture semantic and syntactic relationships
- Learned during model training

### Sequence-to-Sequence Learning
- Maps variable-length input sequences to output predictions
- Enables learning of temporal patterns
- Foundation for generative models

## Contributing

Contributions are welcome! Feel free to:
- Report bugs and issues
- Suggest improvements and new features
- Submit pull requests with enhancements

## License

This project is open source and available under the MIT License.

## Author

Created by happyananya