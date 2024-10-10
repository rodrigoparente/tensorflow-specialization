# Subword Tokenization

Subword tokenization allows the model to break down words into smaller, meaningful units, which can enhance the model's ability to understand and process text. Consider the following Python code.

**Importing Necessary Libraries**

- Start by importing the `keras_nlp` library for advanced NLP tools, including subword tokenization.
- Also, import `tensorflow_datasets` (TFDS) to load the IMDB dataset.

```python
import keras_nlp
import tensorflow_datasets as tfds
```

**Loading the IMDB Dataset**

- Use TFDS to load the IMDB dataset, specifying the data directory and download options.

```python
# Load the IMDB dataset
imdb = tfds.load("imdb_reviews", as_supervised=True, data_dir='./data', download=False)

# Extract train reviews and labels
train_reviews = imdb['train'].map(lambda review, label: review)
train_labels = imdb['train'].map(lambda review, label: label)
```

**Computing Subword Vocabulary**

- Utilize the `compute_word_piece_vocabulary` function from `keras_nlp` to generate the subword vocabulary from the training reviews.
- Set the maximum vocabulary size to 8,000 and reserve tokens for padding and unknown words.

```python
# Compute word piece vocabulary
keras_nlp.tokenizers.compute_word_piece_vocabulary(
    train_reviews,
    vocabulary_size=8000,
    reserved_tokens=["[PAD]", "[UNK]"],
    vocabulary_output_file='imdb_vocab_subwords.txt'
)
```

**Creating the WordPiece Tokenizer**

- Instantiate a `WordPieceTokenizer` using the generated vocabulary file.

```python
# Create a WordPieceTokenizer
subword_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary='./imdb_vocab_subwords.txt'
)
```

**Tokenization Process**

- Use the `tokenized` method to convert strings into integer sequences and the `de-tokenized` method to revert them back to strings.

```python
sample_string = 'TensorFlow, from basics to mastery'
tokenized_string = subword_tokenizer.tokenize(sample_string)
print('Tokenized string is {}'.format(tokenized_string))
```

- The tokenization process results in a higher number of tokens compared to the original number of words due to the use of subwords.

```python
original_string = subword_tokenizer.detokenize(tokenized_string).numpy().decode("utf-8")
print('The original string: {}'.format(original_string))
```

**Analyzing Tokens**

- Print the individual tokens for a sample sentence, which may include special tokens like hashtags indicating the position of the token within a word.

```python
for i in range(len(tokenized_string)):
    subword = subword_tokenizer.detokenize(tokenized_string[i:i+1]).numpy().decode("utf-8")
    print(subword)
```

**Classifying IMDB Reviews**

- The model for classification should be familiar; however, it's important to note the output shape of the vectors from the tokenizer.
- Instead of flattening the output, use Global Average Pooling 1D for better processing.

```python
embedding_dim = 64
MAX_LENGTH = 200  # Assuming this was defined earlier

# Define the model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(MAX_LENGTH,)),
    tf.keras.layers.Embedding(subword_tokenizer.vocabulary_size(), embedding_dim),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

**Training the Model**

- Compile and train the model using standard code patterns.
- Graph the results to visualize performance. The results might show that subword meanings can appear nonsensical when isolated, but they gain meaning when combined in sequences.

```python
# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
num_epochs = 10
history = model.fit(train_dataset,
                    epochs=num_epochs,
                    validation_data=test_data)
```

## Rerences

- [Keras NLP](https://keras.io/keras_nlp/)
- [WordPieceTokenizer](https://keras.io/api/keras_nlp/tokenizers/word_piece_tokenizer/)
- [compute_word_piece_vocabulary](https://keras.io/api/keras_nlp/tokenizers/compute_word_piece_vocabulary/)
- [Fast WordPiece Tokenization](https://arxiv.org/abs/2012.15524)