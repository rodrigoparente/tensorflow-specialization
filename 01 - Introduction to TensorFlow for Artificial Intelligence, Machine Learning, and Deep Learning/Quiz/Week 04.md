# Questions and Answers

## 1. Question 1
Given the choices below, how do you change the images' pixel values into the range 0 to 1?

- [ ] By setting the "normalize" parameter in `tf.keras.utils.image_dataset_from_directory()`.
- [ ] By setting the "rescale" parameter in `tf.keras.utils.image_dataset_from_directory()`.
- [x] By using the `tf.keras.layers.Rescaling()` layer.
- [ ] By using the `tf.keras.layers.Normalization()` layer.
- [ ] TensorFlow automatically does it for you.

**Correct**  
You’ve got it! This is the correct method for rescaling the pixel values.

---

## 2. Question 2
How do you assign labels to images when using `tf.keras.utils.image_dataset_from_directory()`?

- [x] It’s based on the directory the image is contained in.
- [ ] You have to manually do it.
- [ ] It’s based on the file name.
- [ ] TensorFlow figures it out from the contents of each file.

**Correct**  
That’s right! The directory of the image is the label.

---

## 3. Question 3
When you reduce the resolution of the images before training the network, which of the following after effects happen?

- [x] The training is faster.
  
  **Correct**  
  Correct. Because the image is smaller, there are fewer calculations to be done.

- [x] Training results may differ.
  
  **Correct**  
  Correct. The image now contains different information, thus the results may differ.

- [ ] You no longer need to rescale the pixel values.

- [x] You lose some of the information from the original images.

  **Correct**  
  This is correct. If you reduce the image size, you will lose some information because you have fewer pixels to store it in.

---

## 4. Question 4
When you specify the input_shape in `tf.keras.layers.Conv2D()` to be (300, 300, 3), what does that mean?

- [x] Every image will be 300x300 pixels, with 3 bytes to define color.
- [ ] There will be 300 horses and 300 humans, loaded in batches of 3.
- [ ] Every image will be 300x300 pixels, and there should be 3 Convolutional Layers.
- [ ] There will be 300 images, each size 300, loaded in batches of 3.

**Correct**  
Nailed it! `input_shape` specifies image resolution.

---

## 5. Question 5
If your training accuracy is close to 1.000 but the validation accuracy is far from it, what’s the risk here? Select the best answer.

- [ ] You’re overfitting on your validation data.
- [x] You’re overfitting on your training data.
- [ ] No risk, that’s a great result.
- [ ] You’re underfitting on your validation data.

**Correct**  
Great job! The model learned the training data too closely, and may therefore fail to generalize to unseen data.

---

## 6. Question 6
How do you specify the target resolution for the images?

- [ ] By setting the "training_size" parameter in `tf.keras.utils.image_dataset_from_directory()`.
- [ ] By setting the "target_size" parameter in `tf.keras.utils.image_dataset_from_directory()`.
- [x] By setting the "image_size" parameter in `tf.keras.utils.image_dataset_from_directory()`.
- [ ] By using the `tf.keras.layers.Rescaling()` layer.

**Correct**  
That's right!
