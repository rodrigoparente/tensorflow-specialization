# Questions and Answers

## 1. Question 1
If I want to view the history of my training, how can I access it?

- [ ] Use `model.fit` to train the model.
- [ ] Pass the parameter ‘history=true’ to `model.fit`.
- [ ] Download the model and inspect it.
- [x] Create a variable ‘history’ and assign it to the return of `model.fit`.

**Correct**  
Exactly! The `History.history` attribute is a record of training loss values and metrics values at successive epochs.

---

## 2. Question 2
If my image is sized 150x150, and I pass a 3x3 convolution over it, what size is the resulting image? Assume you're using the default settings of the Conv2D layer just like in the lectures.

- [ ] 153x153
- [ ] 150x150
- [x] 148x148
- [ ] 450x450

**Correct**  
Nailed it! Applying a 3x3 convolution would result in a 148x148 image.

---

## 3. Question 3
What does the `image_dataset_from_directory` utility allow you to do? Select the best answer.

- [ ] The ability to easily load images for training.
- [ ] The ability to pick the size of training images.
- [ ] The ability to automatically label images based on their directory name.
- [x] All of the above.

**Correct**  
That's right! It can do all the things mentioned above.

---

## 4. Question 4
When exploring the graphs, the validation accuracy leveled out at about .75 after 2 epochs, but the training accuracy climbed close to 1.0 after 15 epochs. What's the significance of this?

- [ ] There was no point training after 2 epochs, as we overfit to the validation data.
- [x] There was no point training after 2 epochs, as we overfit to the training data.
- [ ] A bigger training set would give us better training accuracy.
- [ ] A bigger validation set would give us better training accuracy.

**Correct**  
Correct! Those values indicate overfitting to the training data.

---

## 5. Question 5
If my data is sized 150x150, and I use Pooling of size 2x2, what size will the resulting image be?

- [ ] 149x149
- [x] 75x75
- [ ] 300x300
- [ ] 148x148

**Correct**  
Nailed it! Applying 2x2 pooling would result in a 75x75 image.

---

## 6. Question 6
What’s the name of the API that allows you to inspect the impact of convolutions on the images?

- [ ] The `model.images` API
- [ ] The `model.convolutions` API
- [x] The `model.layers` API
- [ ] The `model.pools` API

**Correct**  
*(Correct answer not provided in the original text.)*

---

## 7. Question 7
Suppose you want to evaluate a model's performance on unseen data. Why is validation accuracy a better metric than training accuracy?

- [ ] It isn't, they're equally valuable.
- [ ] There's no relationship between them.
- [x] The validation accuracy is based on images that the model wasn't trained on, and thus a better indicator of how the model will perform on new images.
- [ ] The validation dataset is smaller, and thus less accurate at measuring accuracy, so its performance isn't as important.

**Correct**  
*(Correct answer not provided in the original text.)*

---

## 8. Question 8
Why is overfitting more likely to occur on smaller datasets?

- [ ] Because in a smaller dataset, your validation data is more likely to look like your training data.
- [ ] Because there isn't enough data to activate all the convolutions or neurons.
- [ ] Because with less data, the training will take place more quickly, and some features may be missed.
- [x] Because there's less likelihood of all possible features being encountered in the training process.

**Correct**  
Undoubtedly! A smaller size decreases the likelihood that the model will recognize all possible features during training.
