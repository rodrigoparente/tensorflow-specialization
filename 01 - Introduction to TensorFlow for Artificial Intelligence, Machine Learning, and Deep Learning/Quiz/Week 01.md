# Questions and Answers

## 1. Question 1
In the context of machine learning, what is convergence?

- [x] The process of getting very close to the correct answer
- [ ] An analysis that corresponds too closely or exactly to a particular set of data
- [ ] A programming API for AI
- [ ] A dramatic increase in loss

**Correct**  
That’s right! Convergence is when guesses get better and better, closing to a 100% accuracy.

---

## 2. Question 2
What is the difference between traditional programming and machine learning?

- [ ] Machine learning identifies complex activities such as golf, while traditional programming is better suited to simpler activities such as walking.
- [x] In traditional programming, a programmer has to formulate or code rules manually, whereas, in machine learning, the algorithm automatically formulates the rules from the data.

**Correct**  
Exactly! Machine learning algorithms build a model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to do so.

---

## 3. Question 3
What does `model.fit()` do?

- [ ] It makes a model fit the available memory.
- [ ] It optimizes an existing model.
- [x] It trains the neural network to fit the inputs to the expected outputs.
- [ ] It determines if your activity is good for your body.

**Correct**  
Correct! The training takes place using the `.fit()` command.

---

## 4. Question 4
What do we call the process of telling the computer what the data represents (i.e. this data is for walking, this data is for running)?

- [ ] Learning the Data
- [ ] Categorizing the Data
- [ ] Programming the Data
- [x] Labeling the Data

**Correct**  
Yes! Labeling typically takes a set of unlabeled data and augments each piece of it with informative tags.

---

## 5. Question 5
What does the optimizer do?

- [ ] Decides to stop training a neural network when an optimal threshold is reached.
- [x] Updates the weights to decrease the total loss and generate an improved guess.
- [ ] Figures out how to efficiently compile your code to optimize the training.
- [ ] Measures how good the current guess is.

**Correct**  
Nailed it! The optimizer figures out the next guess based on the loss function.

---

## 6. Question 6
What is a Dense layer?

- [ ] A layer of disconnected neurons
- [x] A layer of neurons fully connected to its adjacent layers
- [ ] A single neuron
- [ ] An amount of mass occupying a volume

**Correct**  
Correct! In Keras, dense is used to define this layer of connected neurons.

---

## 7. Question 7
At any time during training, how do you measure how good the current ‘guess’ of the neural network is?

- [ ] Training a neural network
- [ ] Figuring out if you win or lose
- [x] Using the loss function

**Correct**  
Absolutely! An optimization problem seeks to minimize a loss function.

---

## 8. Question 8
When building a TensorFlow Keras model, how do you define the expected shape of the input data?

- [ ] Using a `tf.keras.InputLayer` that specifies the shape of the data via the shape argument
- [ ] Setting the `input_shape` argument of a `tf.keras.layers.Dense` or other first layer your model uses
- [ ] No need to; TensorFlow is capable of inferring this for you
- [x] Using a `tf.keras.Input` that specifies the shape of the data via the shape argument

**Correct**  
Indeed! It is a good practice to define a `tf.keras.Input` as the first element of your neural network so that it has a clearly defined input shape.
