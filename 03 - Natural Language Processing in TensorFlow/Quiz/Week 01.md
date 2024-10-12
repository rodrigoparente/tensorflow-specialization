# Questions and Answers

## 1. Question 1
In the lectures, what is the name of the layer used to generate the vocabulary?

- [ ] Tokenizer
- [ ] TextTokenizer
- [x] TextVectorization
- [ ] WordTokenizer

**Correct**  
That's right!

---

## 2. Question 2
Once you have generated a vocabulary, how do you encode a string sentence to an integer sequence?

- [ ] Pass the string to the get_vocabulary() method.
- [x] Pass the string to the adapted TextVectorization layer.
- [ ] Use the texts_to_tokens() method of the adapted TextVectorization layer.
- [ ] Use the texts_to_sequences() method of the adapted TextVectorization layer.

**Correct**  
That's right!

---

## 3. Question 3
If you have a number of sequences of different lengths, how do you ensure that they are understood when fed into a neural network?

- [ ] Make sure that they are all the same length using the pad_sequences method of the TextVectorization layer.
- [ ] Specify the input layer of the Neural Network to expect different sizes with dynamic_length.
- [x] Use the pad_sequences function from tf.keras.utils.
- [ ] Process them on the input layer of the Neural Network using the pad_sequences property.

**Correct**  
That's right!

---

## 4. Question 4
What happens at encoding when passing a string that is not part of the vocabulary?

- [ ] The word is replaced by the most common token.
- [ ] The word isn’t encoded, and the sequencing ends.
- [x] An out-of-vocabulary token is used to represent it.
- [ ] The word isn’t encoded, and is replaced by a zero in the sequence.

**Correct**  
Correct!

---

## 5. Question 5
When padding sequences, if you want the padding to be at the end of the sequence, how do you do it?

- [x] Pass padding=’post’ to pad_sequences when initializing it.
- [ ] Call the padding method of the pad_sequences object, passing it ‘post’.
- [ ] Pass padding=’after’ to pad_sequences when initializing it.
- [ ] Call the padding method of the pad_sequences object, passing it ‘after’.

**Correct**  
That's right!

---

## 6. Question 6
What's one way to convert a list of strings named 'sentences' to integer sequences? Assume you adapted a TextVectorization layer and assigned it to a variable named 'vectorize_layer'.

- [ ] vectorize_layer.tokenize(sentences)
- [ ] vectorize_layer.fit_to_text(sentences)
- [ ] vectorize_layer.fit(sentences)
- [x] vectorize_layer(sentences)

**Correct**  
Correct!

---

## 7. Question 7
If you have a number of sequences of different length, and call pad_sequences on them, what’s the default result?

- [ ] Nothing, they’ll remain unchanged.
- [x] They’ll get padded to the length of the longest sequence by adding zeros to the beginning of shorter ones.
- [ ] They’ll get cropped to the length of the shortest sequence.
- [ ] They’ll get padded to the length of the longest sequence by adding zeros to the end of shorter ones.

**Correct**  
Correct!

---

## 8. Question 8
Using the default settings, how does the TextVectorization standardize the string inputs?

- [x] By lowercasing and stripping punctuation.
- [ ] By lowercasing the strings.
- [ ] By stripping punctuation.
- [ ] By arranging the strings in alphabetical order.

**Correct**  
That's right! This is the default setting in the 'standardize' parameter of the TextVectorization layer.
