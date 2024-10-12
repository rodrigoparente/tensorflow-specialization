# Questions and Answers

## 1. Question 1
When stacking LSTMs, how do you instruct an LSTM to feed the next one in the sequence?

- [x] Ensure that return_sequences is set to True only on units that feed to another LSTM
- [ ] Ensure that return_sequences is set to True on all units
- [ ] Ensure that they have the same number of units
- [ ] Do nothing, TensorFlow handles this automatically

**Correct**  
Correct!

---

## 2. Question 2
How does an LSTM help understand meaning when words that qualify each other aren’t necessarily beside each other in a sentence?

- [ ] They shuffle the words randomly
- [ ] They don’t
- [ ] They load all words into a cell state
- [x] Values from earlier words can be carried to later ones via a cell state

**Correct**  
Correct!

---

## 3. Question 3
What’s the best way to avoid overfitting in NLP datasets?

- [ ] Use LSTMs
- [ ] Use GRUs
- [ ] Use Conv1D
- [x] None of the above

**Correct**  
Correct!

---

## 4. Question 4
What keras layer type allows LSTMs to look forward and backward in a sentence?

- [ ] Bothdirection
- [ ] Bilateral
- [x] Bidirectional
- [ ] Unilateral

**Correct**  
Correct!

---

## 5. Question 5
Why does sequence make a large difference when determining semantics of language?

- [ ] It doesn’t
- [ ] Because the order in which words appear dictate their meaning
- [x] Because the order in which words appear dictate their impact on the meaning of the sentence
- [ ] Because the order of words doesn’t matter

**Correct**  
Correct!

---

## 6. Question 6
How do Recurrent Neural Networks help you understand the impact of sequence on meaning?

- [ ] They shuffle the words evenly
- [x] They carry meaning from one cell to the next
- [ ] They look at the whole sentence at a time
- [ ] They don’t

**Correct**  
That's right!

---

## 7. Question 7
What’s the output shape of a bidirectional LSTM layer with 64 units?

- [ ] (128,1)
- [ ] (128,None)
- [x] (None, 128)
- [ ] (None, 64)

**Correct**  
That's right!

---

## 8. Question 8
If a sentence has 120 tokens in it, and a Conv1D with 128 filters with a Kernel size of 5 is passed over it, what’s the output shape?

- [ ] (None, 120, 124)
- [ ] (None, 116, 124)
- [ ] (None, 120, 128)
- [x] (None, 116, 128)

**Correct**  
That's right!
