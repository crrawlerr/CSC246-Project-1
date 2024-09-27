# Adarsh Kumar | akumar39@u.rochester.edu


### How to run ###

- Simply extract the zip file and open the src folder in terminal. 
- then enter "python main.py" and it will start trainig and validating both the datasets on both of our models.
- after completing its run, all the accuracy for both the datasets on both the models will get posted which will also creat a folder and 2 model files.
- for some reason, the model files always have the .npz file extension and I was unable to get rid of it while bing able to erad the data. But the data get read from both the model files just fine. 
- after posting the accuracy results, a plot will be extracted via pop-up which will include a graph between accuracy vs. epoch (compute) between all the possible combinations of our models and datasets. For your convenience, there's already a plot saved as an image with "figure_1" filename in the "project1" folder. 

### Writeup ###

Logistic Regression Analysis

Logistic regression is a fundamental classification technique that models the probability of a categorical dependent variable by using a logistic function.
For the given datasets, we implemented a multivariate logistic regression model with softmax outputs and regularization. Our model is trained using stochastic gradient descent.
After training, we get the following results for our Logistic regression model on both datasets:

Logistic Regression Loans Dataset Training Accuracy: 62.7%
Logistic Regression Loans Dataset Validation Accuracy: 62.5%
Logistic Regression Water Dataset Training Accuracy: 92.05%
Logistic Regression Water Dataset Validation Accuracy: 76.2%

Our logistic regression model performed better on the water dataset compared to the loans dataset. It is hard to pinpoint the exact reason as the test data is hidden, but this could be due to the nature of the features and the complexity of the datasets. The water dataset, with its sonar response features, might be more linearly separable than the financial and personal attributes in the loans dataset.

Better performance indicates better accuracy and it is pronounced from our results as the water dataset performs better during both, training and validation for our model. Our model generalizes well on the water dataset but there's a lot of room for improvement on the loans dataset as our accuracy may not be that bad but it's not stellar either (considering 70% accuracy is a good balance).
Our model was trained for 1000 epochs for both datasets. The training process was efficient and did not require extensive computational resources. That could be because I have a quite beefy PC with a 7800X3D CPU but I did notice the fans ramping up while training and validating the model. 
I have run our model multiple times and our model's performance is consistent across all multiple runs, which indicates reliable results.However, its performance on the loans dataset indicates that it may struggle with more complex, non-linear relationships.

The plot of accuracy vs. epoch automatically gets generated after running the "main'py" file and it includes data for all the combinations of datasets and models we have in this project so it is quite easier to see where our models stands on each of the datasets.
I have included a generated plot image file "figure_1" in the "project1" folder for your reference and as you can see, all the plots starts our going higher in accuracy but plateau in the end when going higher and higher in epoch/time. 


Multilayer Perceptron Analysis

A multilayer perceptron (MLP) is a class of feedforward artificial neural networks. It consists of at least three layers: an input layer, a hidden layer, and an output layer. MLPs have more theoretical power than logistic regression but come with greater computational demands and more subtle training requirements.
For the given datasets, we implemented an MLP with 10 tanh units in the hidden layer and softmax outputs. Our model was trained using stochastic gradient descent with dropout regularization.
After training, we get the following results for our MLP model on both datasets:

MLP Loans Dataset Training Accuracy: 59.5%
MLP Loans Dataset Validation Accuracy: 59.9%
MLP Water Dataset Training Accuracy: 92.7%
MLP Water Dataset Validation Accuracy: 76.2%

Our MLP model was trained using gradient descent with a learning rate of 0.01 and 1000 epochs. Regularization was applied to prevent overfitting. The training process involved forward propagation to compute the outputs and backpropagation to update the weights. The tanh activation function was used in the hidden layer to introduce non-linearity.

Our MLP model also performed better on the water dataset compared to the loans dataset. The lower validation accuracy on the water dataset suggests that our MLP model was unable to capture more complex patterns in the loans dataset's data compared to our logistic regression model.

Although our MLP model performed nearly the same for validation on the water dataset as compared to our Logistic Regression model, the results for the loans dataset are very close and comparable but ultimately, just as disappointing. Our MLP model generalizes well on both datasets as the accuracy on both datasets is is obviously worse as compared to our Logistic Regression model accuracy for both datasets.
Our MLP model required more computational resources due to the additional hidden layer and backpropagation process as compared to our Logistic Regression model but the training process was fairly manageable within the given constraints. When it comes to consistency, you will get the worst result on the first run and the results will improve slightly with each run. I am guessing it will only continue to a certain amount of runs but I have not tested that limit. Regardless, the first run is not that bad to begin with and the improvements are very slight so there's not much to worry about here.

The plot for our MLP model is included in the same graph for our Logistic Regression Model.


Research Question

2. 

The way we initialize our neural networks, especially Multilayer Perceptrons (MLPs), is really important for how well they learn. Different ways of initialize can make a big difference in how fast they learn, whether they get stuck in bad spots, and how well they do overall (accuracy, stability). I'll compare different ways of initialize and how they affect these things. It's important to initialize right because if we don't, the gradients (which help the network learn) can either become really tiny and make learning hard, or become really huge and make learning unstable. Also, if we start off with all the weights the same, the network can't tell the neurons apart and ends up learning things it already knows. A right initialization scheme can help the network learn faster and more steadily, while a wrong initialization scheme can make learning slow and wobbly, and might need a lot of extra time or adjustments. Finally, how we initialize directly affects how well the network can learn. If we initialize wrong, the network might not be able to learn as well as it could, even if we train it a lot. But if we initialize right, it's more likely to find a good solution. Let's now discuss and compare some popular initialization schemes:

(i) Zero-mean Gaussian Initialization: It is a weight initialization method where the weights are drawn from a Gaussian (or normal) distribution with a mean of zero and a specified variance. This is one of the simplest and most common initialization methods used historically in neural networks, but like other methods, its effectiveness depends on the network architecture and the activation functions being used.
Based on Zero-mean Gaussian Initialization, our MLP model achieved an accuracy of 59.26% on the loans dataset and 73.81% on the water dataset.

(ii) Xavier Initialization: It is a widely used weight initialization method designed to address the problem of vanishing and exploding gradients, especially in deep neural networks. The central idea behind Xavier initialization is to maintain the variance of the activations and gradients through layers of the network. This helps prevent the gradients from becoming too small (vanishing gradients) or too large (exploding gradients), which would otherwise slow down or destabilize the training process.
Based on Xavier Initialization, our MLP model achieved an accuracy of 61.23% on the loans dataset and 77.12% on the water dataset.

(iii) He Initialization: It is a a weight initialization method designed specifically for networks that use ReLU (Rectified Linear Unit) and its variants (e.g., Leaky ReLU, Parametric ReLU). The key problem that He Initialization solves is the vanishing gradient issue that can occur in deep networks, particularly when using ReLU activations. Unlike sigmoid and tanh, which squash input values to a small range, ReLU activations output zero for negative inputs, and this can lead to neurons "dying" (producing zero output). If weights are not initialized correctly, the gradients during backpropagation can shrink rapidly, leading to vanishing gradients and a slow training process.
He Initialization ensures that the variance of the gradients remains stable as they propagate through the network layers, allowing for deeper networks and faster convergence.
Based on He Initialization, our MLP model achieved an accuracy of 62.56% on the loans dataset and 78.129% on the water dataset.

There are many other Initialization schemes like Random, LeCun, Orthagonal, LSUV, SeLU, etc. but it would be too long to discuss all and honestly, I haven't tried all of them yet so maybe in future if given the opportunity. 

Our results indicate that initialization schemes significantly impact the performance of our MLP model. He initialization provided the best results, followed by Xavier initialization and zero-mean Gaussian initialization. In short, the main importance of choosing a good initialization scheme is that a proper initialization helps in faster convergence and better generalization.