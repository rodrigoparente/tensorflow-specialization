# Introduction to NLP

- This course focuses on building models that handle text data.
- Participants will learn to create models capable of understanding text, trained on labeled datasets.
- The objective is for these models to classify new text based on patterns learned from the training data.
- Processing text data presents unique challenges compared to image data.
- Unlike images, which consist of numerical pixel values, text requires different handling methods.
- Key questions include how to effectively process and classify sentences and words using neural networks.

## Word-Based Encodings

- Character encodings, such as ASCII, assign numerical values to individual characters.
- For instance, encoding the words "SILENT" and "LISTEN" using ASCII values would result in the following:

| Letter | Value | Letter | Value |
|--------|-------|--------|-------|
| S      | 083   | L      | 076   |
| I      | 073   | I      | 073   |
| L      | 076   | S      | 083   |
| E      | 069   | T      | 084   |
| N      | 078   | E      | 069   |
| T      | 084   | N      | 078   |

- This example illustrates that "silent" and "listen" contain the same letters but have different meanings, highlighting the challenge of semantic understanding in neural networks.

**Proposed Approach**

- Instead of relying solely on character encodings, we propose treating words as units of meaning.
- Assigning unique values to words can help train neural networks by providing consistent representations for each word.
- For example, the sentence "I love my dog" can be encoded as follows:

| Word  | Value |
|-------|-------|
| I     | 1     |
| love  | 2     |
| my    | 3     |
| dog   | 4     |

- The entire sentence is then represented as the sequence **1234**.
- When encountering a new sentence, such as "I love my cat," the encoding would be:

| Word  | Value |
|-------|-------|
| I     | 1     |
| love  | 2     |
| my    | 3     |
| cat   | 5     |

- In this case, the words "I," "love," and "my" retain their previously assigned values, while a new token is created for "cat," assigned the value **5**.
- This results in the encoding **1235** for "I love my cat."
- The encodings for both sentences reveal a structural similarity, demonstrating a potential pattern.

## Using APIs

- The goal is to encode sentences into a format that a neural network can process using the **text vectorization layer** in TensorFlow and Keras.
- This layer simplifies the encoding process by generating a vocabulary and converting sentences into vectors.
- It automatically handles various preprocessing tasks, such as:
    - Lowercasing words (e.g., "I" becomes "i").
    - Stripping out punctuation (e.g., "dog!" is treated as "dog").

**Implementation Steps**

- the sentences are stored in an array:

```python
sentences = [
    'I love my dog',
    'I love my cat'
]
```
- An instance of the text vectorization layer is created:

```python
vectorize_layer = tf.keras.layers.TextVectorization()
```

- The `adapt` method is called with the sentences, generating the vocabulary based on the input data:

```python
vectorize_layer.adapt(sentences)
```

- The vocabulary can be retrieved using the `get_vocabulary` method. The parameter `include_special_tokens` is set to `False` to exclude additional tokens like empty strings or unknown tokens:

```python
vocabulary = vectorize_layer.get_vocabulary(include_special_tokens=False)
```

- If you `print` the `vocabulary`, you will end up with:

```python
["i", "love", "my", "dog", "cat"]
```

- If, instead of setting `include_special_tokens=False`, you set it to `True`, your `vocabulary` would be:

```python
["", "[UNK]", "i", "love", "my", "dog", "cat"]
```

## Text to Sequence

- Neural networks require input data with consistent dimensions, whether working with images or text.
- Sentences, which can vary in length, need to be standardized to the same length before being processed by the network.
- This is similar to resizing images to a uniform size before feeding them into a neural network.
- In text processing, we typically add **padding** to shorter sentences so they match the length of the longest sentence.

**Post-Padding with TensorFlow**

In the following example, we'll see how TensorFlow's `TextVectorization` layer automatically pads shorter sentences with zeros:

```python
import tensorflow as tf

# List of sentences
sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

# Create a TextVectorization layer
vectorize_layer = tf.keras.layers.TextVectorization()

# Adapt the layer to the sentences (learn the vocabulary)
vectorize_layer.adapt(sentences)

# Convert the sentences into sequences of tokenized values
sequences = vectorize_layer(sentences)

print(sequences)
```

This would result in the following output, where shorter sentences are **post-padded** with zeros (`0`) to match the length of the longest sentence (7 tokens):

```bash
tf.Tensor(
    [[ 6, 3, 2, 4, 0, 0, 0],  
     [ 6, 3, 2, 10, 0, 0, 0],  
     [ 6, 3, 2, 4, 0, 0, 0],   
     [ 9, 5, 7, 2, 4, 8, 11]], shape=(4, 7), dtype=tf.int64)
```

**Pre-Padding with `pad_sequences`**

In some cases, you may want to apply **pre-padding**, where the padding is applied to the front of the sentence instead of the back. While the `TextVectorization` layer automatically applies post-padding, you can manually control this using Keras' `pad_sequences` function from `tf.keras.preprocessing.sequence`.

Here’s how you can create pre-padded sequences:

1. Convert your sentences into a TensorFlow dataset and apply vectorization without automatic padding.
2. Use Keras' `pad_sequences` function to apply **pre-padding**.

Here’s an example:

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Convert sentences into a dataset
dataset = tf.data.Dataset.from_tensor_slices(sentences)

# Map the vectorization function to the dataset
vectorized_sentences = dataset.map(lambda x: vectorize_layer(x))

# Convert the dataset to sequences (this will output unpadded sequences)
sequences = list(vectorized_sentences.as_numpy_iterator())

# Apply pre-padding
pre_padded_sequences = pad_sequences(sequences, padding='pre')

# Print pre-padded sequences
print(pre_padded_sequences)
```

The result would be sequences where padding (`0`s) appears at the beginning of shorter sequences, ensuring all sequences have the same length:

```bash
[[ 0, 0, 0,  6,  3,  2,  4],  # 'I love my dog' (pre-padded)
 [ 0, 0, 0,  6,  3,  2, 10],  # 'I love my cat' (pre-padded)
 [ 0, 0, 0,  6,  3,  2,  4],  # 'You love my dog!' (pre-padded)
 [ 9,  5,  7,  2,  4,  8, 11]] # 'Do you think my dog is amazing?' (no padding needed)
```

**Handling Ragged Tensors (Variable-Length Sequences)**

If you prefer to avoid padding altogether during the vectorization step, you can configure the `TextVectorization` layer to output **ragged tensors**. Ragged tensors allow each sequence to retain its original length, making it more flexible for certain tasks.

Here’s how you can enable ragged tensors:

```python
# Create a TextVectorization layer with ragged tensor output
vectorize_layer = tf.keras.layers.TextVectorization(output_mode='int', ragged=True)

# Adapt the layer to the sentences
vectorize_layer.adapt(sentences)

# Apply the vectorization without padding
ragged_sequences = vectorize_layer(sentences)

# Print the ragged sequences
print(ragged_sequences)
```

This will give you sequences with their original lengths:

```bash
<tf.RaggedTensor [[6, 3, 2, 4], [6, 3, 2, 10], [6, 3, 2, 4], [9, 5, 7, 2, 4, 8, 11]]>
```

You can apply padding later using the `pad_sequences` function if needed.

## Sarcasm, really?

- In real-world scenarios, much larger datasets are typically used.
- This lesson we will use a public dataset created by Rashabh Misra and posted on [Kaggle](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/).

**Dataset Structure**

The dataset initially comes as a list of dictionaries, each containing:
- `article_link`: URL of the article.
- `headline`: Text of the headline.
- `is_sarcastic`: Sarcasm label (1 or 0).

**Data Preparation in Python**

To load the dataset into Python and prepare it for tokenization, you can execute the following Python code:

```python
import json

# Load the dataset
with open('path_to_your_file.json', 'r') as f:
    datastore = json.load(f)

# Initialize lists to store the elements
sentences = []
labels = []
urls = []

# Iterate through the data
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])  # Not needed for this task, but included
```

Once the data is prepared in this way, it can be easily passed to a tokenizer and used to train a neural network.