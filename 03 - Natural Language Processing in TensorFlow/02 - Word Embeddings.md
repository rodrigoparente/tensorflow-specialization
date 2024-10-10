## Word Embeddings

- Word embeddings is a process that converts words into vectors in a multi-dimensional space.
- Words with similar meanings or associations are clustered together.
- The process is similar to how features are extracted from images, but for text.

## Embedding Projector

- The [embedding projector](https://projector.tensorflow.org) visualizes high-dimensional data, like word embeddings, by reducing them to 2D or 3D using **PCA** or **t-SNE**.
- It enables interactive exploration, revealing clusters of similar items (e.g., words with similar meanings).
- This helps in understanding data relationships and interpreting machine learning models.
- For instance, "boring" appears in a **negative cluster** with words like "unwatchable," while "fun" shows up in a **positive cluster** with words like "funny."

## TensorFlow Data Services (TFDS)**

- The **[TFDS library](https://www.tensorflow.org/datasets/catalog)** offers a diverse collection of datasets from various domains.
- These include popular **image datasets** (e.g., **MNIST**, **ImageNet**) and **text datasets** (e.g., **IMDB Movie Reviews**, **CNN News Articles**).

## IMDB Reviews Dataset

Start by **loading the IMDB Reviews dataset** using TensorFlow Datasets. This fetches the dataset and its associated metadata.
  
```python
import tensorflow as tf
import tensorflow_datasets as tfds

# Load the IMDB reviews dataset
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
```

The dataset contains **50,000 movie reviews**, split into **25,000 for training** and **25,000 for testing**. The reviews are labeled as either positive (1) or negative (0). Next, you split the data into training and test sets.

```python
# Split the dataset into train and test
train_data, test_data = imdb['train'], imdb['test']
```

Now, **separate the reviews and labels**. This can be done using the **`map()`** function to extract reviews into one variable and labels into another.

```python
# Separate reviews and labels
train_reviews = train_data.map(lambda review, label: review)
train_labels = train_data.map(lambda review, label: label)

test_reviews = test_data.map(lambda review, label: review)
test_labels = test_data.map(lambda review, label: label)
```

To transform the text into numerical sequences, create a **Text Vectorization layer**. This layer converts raw text into tokens. You can set a limit on the number of tokens with the **`max_tokens`** parameter, which here is set to 10,000.

```python
# Create and adapt the vectorization layer
vectorize_layer = tf.keras.layers.TextVectorization(max_tokens=10000)
vectorize_layer.adapt(train_reviews)
```

A **padding function** is necessary to ensure that all sequences have the same length. This function first creates a ragged tensor (sequences of varying lengths), then pads each sequence to a consistent length.

```python
# Define padding function
def padding_func(sequences):
    sequences = sequences.ragged_batch(batch_size=sequences.cardinality())
    sequences = sequences.get_single_element()

    padded_sequences = tf.keras.utils.pad_sequences(sequences.numpy(), maxlen=120, truncating='post', padding='pre')
    padded_sequences = tf.data.Dataset.from_tensor_slices(padded_sequences)

    return padded_sequences
```

Now, apply the vectorization and padding to both the training and test datasets to transform the text into sequences.

```python
# Apply vectorization and padding
train_sequences = train_reviews.map(lambda text: vectorize_layer(text)).apply(padding_func)
test_sequences = test_reviews.map(lambda text: vectorize_layer(text)).apply(padding_func)
```

Finally, **combine the sequences with the labels** and apply caching, shuffling, prefetching, and batching to prepare the dataset for training.

```python
# Combine sequences and labels
train_dataset_vectorized = tf.data.Dataset.zip((train_sequences, train_labels))
test_dataset_vectorized = tf.data.Dataset.zip((test_sequences, test_labels))

# Set up dataset parameters
SHUFFLE_BUFFER_SIZE = 1000
PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE
BATCH_SIZE = 32

# Prepare final datasets
train_dataset_final = (train_dataset_vectorized
    .cache()
    .shuffle(SHUFFLE_BUFFER_SIZE)
    .prefetch(PREFETCH_BUFFER_SIZE)
    .batch(BATCH_SIZE)
)

test_dataset_final = (test_dataset_vectorized
    .cache()
    .prefetch(PREFETCH_BUFFER_SIZE)
    .batch(BATCH_SIZE)
)
```

By combining these steps, you load, process, and prepare the IMDB dataset for use in machine learning tasks like text classification.

## Defining a DNN Model

- Words are represented as vectors in a higher-dimensional space, such as 16 dimensions. Each word's position in this space is determined by its meaning and context.
- This representation allows the model to group words that have similar meanings and sentiments close together based on their vector values. For example, words like "happy" and "joyful" would be located near each other.
- During training, the neural network learns to associate these word vectors with specific labels, such as sentiment. For instance, words commonly found in positive reviews will be linked to positive sentiments.
- As a result of this learning process, the model generates embeddings. These embeddings are 2D arrays where one dimension represents the length of the input sentence, and the other dimension corresponds to the embedding size (e.g., 16).
- In this context, the first dimension of the array reflects how many words are in the sentence, while the second dimension indicates the size of the embedding, capturing the word's meaning in that high-dimensional space.

**Model Structure Example**

Consider the following Python code:

```python
model = tf.keras.Sequential([
    tf.keras.Input(shape=(120,)),  
    tf.keras.layers.Embedding(vocab_size, embedding_dim),  
    tf.keras.layers.Flatten(),  
    tf.keras.layers.Dense(6, activation='relu'),  
    tf.keras.layers.Dense(1, activation='sigmoid')  
])
```

- The `Embedding` layer converts each word into a dense vector of a specified size (`embedding_dim`), while the `Flatten` layer reshapes this output for the subsequent dense layers.
- The dense layers then process these flattened vectors, ultimately producing a binary classification output.
- Alternativelly, instead of using a standard flatten layer, you can opt for the `GlobalAveragePooling1D` layer, which can improve processing efficiency by averaging the embeddings across the sequence.

## Training a DNN Model

To begin, compile your model with the necessary loss function and optimizer. You can also print the model summary to review its architecture.

```python
# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()  # Display the model architecture
```

Next, train the model using your training dataset. Specify the number of epochs and include your validation dataset to assess performance.

```python
# Train the model
num_epochs = 10
model.fit(train_dataset_final, epochs=num_epochs, validation_data=test_dataset_final)
```

Once training is complete, extract the weights from the embedding layer. This layer typically represents words in a dense vector format, allowing the model to learn relationships between words based on their usage in the training data.

```python
# Extract embedding weights
embedding_layer = model.layers[0]  # Accessing the first layer (embedding layer)
embedding_weights = embedding_layer.get_weights()[0]  # Retrieve the weights of the embedding layer
print(embedding_weights.shape)  # Print the shape of the embeddings (e.g., (vocab_size, embedding_dim))
```

In this case, the shape of the embedding weights could be (10,000, 16), indicating that there are 10,000 words in the corpus, each represented as a 16-dimensional vector.

## Saving and Visualizing Embeddings

To visualize the embeddings, save them into two TSV files:

- A **metadata file** containing the words.
- A **vectors file** containing the corresponding embedding values.

```python
# Save embeddings and vocabulary to TSV files for visualization
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')

vocabulary = vectorize_layer.get_vocabulary()  # Obtain the vocabulary from the vectorization layer

for word_num in range(1, len(vocabulary)):
    word_name = vocabulary[word_num]
    word_embedding = embedding_weights[word_num]
    out_m.write(word_name + "\n")  # Write the word to the metadata file
    out_v.write('\t'.join([str(x) for x in word_embedding]) + "\n")  # Write the embedding to the vectors file

out_v.close()  # Close the vectors file
out_m.close()  # Close the metadata file
```

After saving the files, you can use the [TensorFlow Embedding Projector](https://projector.tensorflow.org/) to load the data. This tool allows you to visualize the embeddings in a 3D space, making it easier to explore relationships and clusters among the words.
