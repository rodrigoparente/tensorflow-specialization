# Questions and Answers

## 1. Question 1
In natural language processing, predicting the next item in a sequence is a classification problem. Therefore, after creating inputs and labels from the subphrases, we one-hot encode the labels. What function do we use to create one-hot encoded arrays of the labels?

- [ ] tf.keras.utils.img_to_array
- [ ] tf.keras.utils.SequenceEnqueuer
- [ ] tf.keras.preprocessing.text.one_hot
- [x] tf.keras.utils.to_categorical

**Correct**  
Nailed it!

---

## 2. Question 2
What is a major drawback of word-based training for text generation instead of character-based generation?

- [ ] Character based generation is more accurate because there are less characters to predict
- [ ] Word based generation is more accurate because there is a larger body of words to draw from
- [x] Because there are far more words in a typical corpus than characters, it is much more memory intensive
- [ ] There is no major drawback, it’s always better to do word-based training

**Correct**  
Correct!

---

## 3. Question 3
What are the critical steps in preparing the input sequences for the prediction model?

- [ ] Converting the seed text to a token sequence using texts_to_sequences.

- [x] Pre-padding the subphrases sequences.  
  **Correct**  
  You've got it!

- [x] Generating subphrases from each line using n_gram_sequences.  
  **Correct**  
  Keep it up!

- [ ] Splitting the dataset into training and testing sentences.

---

## 4. Question 4
When predicting words to generate poetry, the more words predicted the more likely it will end up gibberish. Why?

- [x] Because the probability that each word matches an existing phrase goes down the more words you create
- [ ] Because the probability of prediction compounds, and thus increases overall
- [ ] It doesn’t, the likelihood of gibberish doesn’t change
- [ ] Because you are more likely to hit words not in the training set

**Correct**  
That's right!

---

## 5. Question 5
True or False: When building the model, we use a sigmoid activated Dense output layer with one neuron per word that lights up when we predict a given word.

- [ ] True
- [x] False

**Correct**  
Absolutely!
