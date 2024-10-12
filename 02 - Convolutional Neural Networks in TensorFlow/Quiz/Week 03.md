# Questions and Answers

## 1. Question 1
If I put a dropout parameter of 0.2, how many nodes will I lose?

- [x] 20% of them
- [ ] 2% of them
- [ ] 20% of the untrained ones
- [ ] 2% of the untrained ones

**Correct**  
Spot on!

---

## 2. Question 2
How do you change the number of classes the model can classify when using transfer learning? (i.e., the original model handled 1000 classes, but yours handles just 2)

- [ ] Ignore all the classes above yours (i.e., Numbers 2 onwards if I'm just classing 2).
- [ ] Use all classes but set their weights to 0.
- [x] When you add your DNN at the bottom of the network, you specify your output layer with the number of classes you want.
- [ ] Use dropouts to eliminate the unwanted classes.

**Correct**  
Good job!

---

## 3. Question 3
Which is the correct line of code for declaring a dropout of 20% of neurons using TensorFlow?

- [ ] `tf.keras.layers.Dropout(20)`
- [ ] `tf.keras.layers.DropoutNeurons(20)`
- [x] `tf.keras.layers.Dropout(0.2)`
- [ ] `tf.keras.layers.DropoutNeurons(0.2)`

**Correct**  
You've got it!

---

## 4. Question 4
Why do dropouts help avoid overfitting?

- [x] Because neighbor neurons can have similar weights, and thus can skew the final training.
- [ ] Having less neurons speeds up training.

**Correct**  
That's right!

---

## 5. Question 5
Why is transfer learning useful?

- [ ] Because I can use all of the data from the original training set.
- [ ] Because I can use all of the data from the original validation set.
- [x] Because I can use the features that were learned from large datasets that I may not have access to.
- [ ] Because I can use the validation metadata from large datasets that I may not have access to.

**Correct**  
Exactly!

---

## 6. Question 6
Can you use image augmentation with transfer learning models? 

- [ ] No, because you are using pre-set features.
- [x] Yes, you can use image augmentation when training the layers you added to the pre-trained model.

**Correct**  
That's right!

---

## 7. Question 7
How did you lock or freeze a layer from retraining?

- [ ] `tf.freeze(layer)`
- [ ] `tf.layer.frozen = True`
- [ ] `tf.layer.locked = True`
- [x] `layer.trainable = False`

**Correct**  
Well done!

---

## 8. Question 8
The dropout rate determines how many neurons are removed from the network during training. Which of the two cases below do you think will happen if it is set too high?

- [x] The network would lose specialization to the effect that it would be inefficient or ineffective at learning. 
- [ ] Training time would increase due to the extra calculations being required for higher dropout.

**Correct**  
Indeed! This will drive accuracy down.
