# Questions and Answers

## 1. Question 1
What is the purpose of a test set? Select the best answer.

- [ ] To make testing quicker
- [ ] To make training quicker.
- [ ] To train a network with previously unseen data
- [x] To see how well the model does on previously unseen data.

**Correct**  
Nailed it! You can use a test set to evaluate a model's performance on unseen data.

---

## 2. Question 2
What does the ReLU activation function do?

- [x] It returns x if x is greater than zero, otherwise it returns zero.
- [ ] It returns the negative of x.
- [ ] It returns x if x is less than zero, otherwise it returns zero.
- [ ] For a value x, it returns 1/x.

**Correct**  
Correct! The rectifier or ReLU (Rectified Linear Unit) activation function returns x if x is greater than zero.

---

## 3. Question 3
What is the resolution and color format of the Fashion MNIST dataset?

- [x] 28x28 grayscale
- [ ] 82x82 grayscale
- [ ] 100x100 RGB
- [ ] 28x28 RGB

**Correct**  
Spot on!

---

## 4. Question 4
How do you specify a callback function that activates during training?

- [ ] You pass it to the callbacks parameter in the .compile() method.
- [ ] You define it as one of the layers in your model.
- [x] You pass it to the callbacks parameter of the .fit() method.

**Correct**  
That’s right!

---

## 5. Question 5
True or False: You can use the on_epoch_end method to get the current state of training at the end of every epoch.

- [ ] False
- [x] True

**Correct**  
Absolutely! It activates at the end of every epoch.

---

## 6. Question 6
Why are there 10 output neurons in the Fashion MNIST computer vision model?

- [ ] To make it classify 10x faster.
- [x] There are 10 different labels.
- [ ] Purely arbitrary.
- [ ] To make it train 10x faster.

**Correct**  
Exactly! There are 10 output neurons because Fashion MNIST has 10 classes of clothing in the dataset. These should always match.
