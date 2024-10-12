# Questions and Answers

## 1. Question 1
When using image augmentation with `image_dataset_from_directory`, what happens to your raw image data on-disk?

- [ ] A copy will be made, and the copies are augmented.
- [ ] A copy will be made, and the originals will be augmented.
- [x] Nothing.
- [ ] The images will be edited on disk, so be sure to have a backup.

**Correct**  
That is, in fact, true. Nothing happens.

---

## 2. Question 2
What layer is used to convert image pixel values from the range [0, 255] to [0, 1]? 

- [ ] Conversion.
- [ ] Translation.
- [ ] Resize.
- [x] Rescaling.

**Correct**  
That's right!

---

## 3. Question 3
The diagram for traditional programming had Rules and Data in, but what came out?

- [x] Answers.
- [ ] Binary.
- [ ] Machine Learning.
- [ ] Bugs.

**Correct**  
Exactly! Machine learning algorithms build a model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to do so.

---

## 4. Question 4
When training for multiple classes, what is the parameter you specify in `image_dataset_from_directory` if you like to label them for categorical_crossentropy loss?

- [ ] `label_mode='int'`
- [ ] `class_mode='int'`
- [x] `label_mode='categorical'`
- [ ] `class_mode='categorical'`

**Correct**  
Nicely done!

---

## 5. Question 5
Can you use image augmentation with transfer learning? 

- [ ] No - [ ] because the layers are frozen so they can't be augmented.
- [x] Yes. It's pre-trained layers that are frozen. So you can augment your images as you train the trainable layers of the DNN with them.

**Correct**  
You've got it!

---

## 6. Question 6
Applying convolutions on top of a DNN will have what impact on training?

- [ ] It will be slower.
- [ ] It will be faster.
- [ ] There will be no impact.
- [x] It depends on many factors. It might make your training faster or slower, and a poorly designed convolutional layer may even be less efficient than a plain DNN!

**Correct**  
Exactly!

---

## 7. Question 7
What is a convolution? 

- [ ] A technique to make images smaller.
- [ ] A technique to make images larger.
- [x] A technique to extract features from an image.
- [ ] A technique to remove unwanted images.

**Correct**  
You've got it!

---

## 8. Question 8
Why does the DNN for Fashion MNIST have 10 output neurons?

- [ ] To make it train 10x faster.
- [ ] To make it classify 10x faster.
- [ ] Purely arbitrary.
- [x] The dataset has 10 classes.

**Correct**  
Exactly! There are 10 output neurons because we have 10 classes of clothing in the dataset. These should always match.
