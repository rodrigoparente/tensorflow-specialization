# Questions and Answers

## 1. Question 1
When using image augmentation with `image_dataset_from_directory`, what happens to your raw image data on-disk?

- [ ] It gets overwritten, so be sure to make a backup.
- [ ] A copy is made and the augmentation is done on the copy.
- [x] Nothing, all augmentation is done in-memory.
- [ ] It gets deleted.

**Correct**  
That's right!

---

## 2. Question 2
How does image augmentation help solve overfitting?

- [ ] It slows down the training process.
- [x] It manipulates the training set to generate more scenarios for features in the images.
- [ ] It manipulates the validation set to generate more scenarios for features in the images.
- [ ] It automatically fits features to images by finding them through image processing techniques.

**Correct**  
That's right!

---

## 3. Question 3
True or False: Using image augmentation effectively simulates having a larger variation of images in the training dataset.

- [ ] False
- [x] True

**Correct**  
Exactly!

---

## 4. Question 4
When using image augmentation, model training gets...

- [x] slower
- [ ] faster
- [ ] stays the same
- [ ] much faster

**Correct**  
That's right!

---

## 5. Question 5
If my training data only has people facing left, but I want to classify people facing right, how would I avoid overfitting?

- [ ] Use the ‘flip’ parameter of `image_dataset_from_directory`.
- [ ] Use the 'flip' parameter of `image_dataset_from_directory` and set 'horizontal'.
- [ ] Use the `RandomFlip` layer and set `mode='vertical'`.
- [x] Use the `RandomFlip` layer and set `mode='horizontal'`.

**Correct**  
That's right!

---

## 6. Question 6
How do you use image augmentation in TensorFlow?

- [ ] With the `keras.augment` API.
- [ ] With the `tf.augment` API.
- [ ] You have to write a plugin to extend `tf.layers`.
- [x] Using preprocessing layers from the Keras Layers API.

**Correct**  
That's right!

---

## 7. Question 7
After adding data augmentation and using the same batch size and steps per epoch, you noticed that each training epoch became a little slower than when you trained without it. Why?

- [x] Because the image preprocessing takes cycles.
- [ ] Because the augmented data is bigger.
- [ ] Because the training is making more mistakes.
- [ ] Because there is more data to train on.

**Correct**  
That's right! It will take some time to generate and load the additional images into memory.

---

## 8. Question 8
What does the `fill_mode` parameter do?

- [ ] There is no `fill_mode` parameter.
- [ ] It creates random noise in the image.
- [x] It attempts to recreate lost information after a transformation like a shear.
- [ ] It masks the background of an image.

**Correct**  
That's right!
