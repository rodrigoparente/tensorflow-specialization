# Questions and Answers

## 1. Question 1
To use word embeddings in TensorFlow, in a Sequential model, what is the name of the layer?

- [ ] tf.keras.layers.WordEmbedding
- [x] tf.keras.layers.Embedding
- [ ] tf.keras.layers.Word2Vector
- [ ] tf.keras.layers.Embed

**Correct**  
That's right!

---

## 2. Question 2
Using the default settings, what does the 'max_tokens' parameter do when initializing the TextVectorization layer?

- [ ] It specifies the maximum size of the vocabulary, and picks the most common ‘max_tokens - [ ] 1’ words
- [ ] It specifies the maximum size of the vocabulary, and picks the most common ‘max_tokens’ words
- [ ] It errors out if there are more than max_tokens distinct words in the corpus
- [x] It specifies the maximum size of the vocabulary, and picks the most common ‘max_tokens - [ ] 2’ words

**Correct**  
That's right! This includes the 0 index for padding and 1 for out-of-vocabulary words.

---

## 3. Question 3
What is the name of the TensorFlow library containing common data that you can use to train and test neural networks?

- [x] TensorFlow Datasets
- [ ] There is no library of common data sets, you have to use your own
- [ ] TensorFlow Data Libraries
- [ ] TensorFlow Data

**Correct**  
Correct!

---

## 4. Question 4
What is the purpose of the embedding dimension?

- [ ] It is the number of dimensions required to encode every word in the corpus
- [ ] It is the number of words to encode in the embedding
- [x] It is the number of dimensions for the vector representing the word encoding
- [ ] It is the number of letters in the word, denoting the size of the encoding

**Correct**  
That's right!

---

## 5. Question 5
IMDB Reviews are either positive or negative. What type of loss function should be used in this scenario?

- [ ] Categorical crossentropy
- [x] Binary crossentropy
- [ ] Binary Gradient descent
- [ ] Adam

**Correct**  
Correct!

---

## 6. Question 6
How are the labels for the IMDB dataset encoded?

- [x] Reviews encoded as a number 0-1
- [ ] Reviews encoded as a number 1-5
- [ ] Reviews encoded as a number 1-10
- [ ] Reviews encoded as a boolean true/false

**Correct**  
Correct!

---

## 7. Question 7
How many reviews are there in the IMDB dataset and how are they split?

- [ ] 60,000 records, 50/50 train/test split
- [x] 50,000 records, 50/50 train/test split
- [ ] 60,000 records, 80/20 train/test split
- [ ] 50,000 records, 80/20 train/test split

**Correct**  
That's right!
