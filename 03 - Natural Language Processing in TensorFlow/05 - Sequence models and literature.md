# Text Classification vs. Text Generation

- Now, the focus shifts to generating new text, which may seem novel but builds on concepts already covered.
- Generating text is framed as a prediction problem, similar to how a neural network predicts images.
- For text, the network is trained to predict the next word in a sequence, just as it predicts image content based on pixels.

**Training Data Preparation**

- A body of text is used to create a dataset by extracting the vocabulary.
- The input $ x $ consists of a sequence of words or phrases, and the next word in that sequence becomes the target output $ y $.
- Example: In the phrase “twinkle twinkle little star”:
    - The input $ x $ would be "twinkle twinkle little."
    - The target $ y $ would be "star."
- By training on a large corpus of text, the network can generate sophisticated text sequences based on the learned patterns.

## Looking Into Code

The first step is to import necessary libraries. TensorFlow is used to build the neural network, while NumPy is used for handling numerical data.

```python
import tensorflow as tf
import numpy as np
```

The poem is stored as a string. It's the raw data that will be used to train the model. Line breaks in the text are represented by `\n`.

```python
data = "In the town of Athy one Jeremy Lanigan\nBattered away til he hadnt a pound.\nHis father died and made him a man again\nLeft him a farm and ten acres of ground.\n\nHe gave a grand party for friends and relations\nWho didnt forget him when come to the wall,\nAnd if youll but listen Ill make your eyes glisten\nOf the rows and the ructions of Lanigan's Ball.\n\nMyself to be sure got free invitation,\nFor all the nice girls and boys I might ask,\nAnd just in a minute both friends and relations\nWere dancing round merry as bees round a cask.\n\nJudy ODaly, that nice little milliner,\nShe tipped me a wink for to give her a call,\nAnd I soon arrived with Peggy McGilligan\nJust in time for Lanigans Ball."
```

The text is converted to lowercase for consistency and split into individual lines using `\n`. This breaks the poem into manageable pieces, and each line will be treated as a separate sequence for model training.

```python
corpus = data.lower().split("\n")
```

A `TextVectorization` layer is created, which will transform each word in the corpus into a corresponding integer (a token). The `adapt()` function learns the mapping from the corpus.

```python
vectorize_layer = tf.keras.layers.TextVectorization()
vectorize_layer.adapt(corpus)
```

The model's vocabulary (the list of words it knows) is extracted from the vectorization layer. The vocabulary size is important for defining the input and output layers of the neural network.

```python
vocabulary = vectorize_layer.get_vocabulary()
vocab_size = len(vocabulary)
```

For each line in the corpus, sequences of increasing length are generated. For example, for the line "Jeremy Lanigan", the sequence will be split into multiple inputs like `["Jeremy"]`, `["Jeremy", "Lanigan"]`. These will later be used to predict the next word.

```python
input_sequences = []
for line in corpus:
    sequence = vectorize_layer(line).numpy()
    for i in range(1, len(sequence)):
        n_gram_sequence = sequence[:i+1]
        input_sequences.append(n_gram_sequence)
```

Since neural networks require input data of the same length, shorter sequences are padded with zeros at the beginning to match the longest sequence length.

```python
max_sequence_len = max([len(x) for x in input_sequences])

input_sequences = np.array(tf.keras.utils.pad_sequences(input_sequences,
                                                        maxlen=max_sequence_len,
                                                        padding='pre'))
```

The input sequences (`xs`) contain all but the last word, and the output labels (`ys`) are the last word of each sequence. This setup is used to train the model to predict the next word in a sequence.

```python
xs = input_sequences[:,:-1]
labels = input_sequences[:,-1]

ys = tf.keras.utils.to_categorical(labels, num_classes=vocab_size)
```

The neural network model consists of:
    - An `Embedding` layer that converts word indices into dense vectors of a fixed size.
    - An `LSTM` (Long Short-Term Memory) layer that processes the sequence of word embeddings and retains information across the sequence.
    - A `Dense` layer with a softmax activation function to predict the next word in the sequence from the vocabulary.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(max_sequence_len - 1,)),
    tf.keras.layers.Embedding(vocab_size, 64),
    tf.keras.layers.LSTM(20),
    tf.keras.layers.Dense(vocab_size, activation='softmax')
])
```

The model is compiled with categorical cross-entropy as the loss function and Adam optimizer. Accuracy is used as the evaluation metric.

```python
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

The model is trained for 500 epochs. During each epoch, it processes the input sequences (`xs`) and learns to predict the next word (`ys`).

```python
history = model.fit(xs, ys, epochs=500)
```

To generate new text, a seed phrase is provided. The model predicts the next word in the sequence, and the process is repeated for the desired number of words. In this case, 10 additional words are generated based on the seed text "Laurence went to Dublin."

```python
seed_text = 'Laurence went to Dublin'
next_words = 10

for _ in range(next_words):
    sequence = vectorize_layer(seed_text)
    sequence = tf.keras.utils.pad_sequences(
        [sequence],
        padding='pre',
        maxlen=max_sequence_len - 1)

    probabilities = model.predict(sequence, verbose=0)
    predicted = np.argmax(probabilities, axis=-1)[0]
    output_word = vocabulary[predicted]
    seed_text += ' ' + output_word

print(seed_text)
```