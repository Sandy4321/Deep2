## CS294A/CS294W Softmax Exercise 

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  softmax exercise. You will need to write the softmax cost function 
#  in softmaxCost.m and the softmax prediction function in softmaxPred.m. 
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#  (However, you may be required to do so in later exercises)

##======================================================================
## STEP 0: Initialise constants and parameters
#
#  Here we define and initialise some constants which allow your code
#  to be used more generally on any arbitrary input. 
#  We also initialise some parameters used for tuning the model.

inputSize = 28*28; # Size of input vector (MNIST images are 28x28)
numClasses = 10;     # Number of classes (MNIST images fall into 10 classes)

lambda = 1e-4; # Weight decay parameter

##======================================================================
## STEP 1: Load data
#
#  In this section, we load the input and output data.
#  For softmax regression on MNIST pixels, 
#  the input data is the images, and 
#  the output data is the labels.
#

# Change the filenames if you've saved the files under different names
# On some platforms, the files might be saved as 
# train-images.idx3-ubyte / train-labels.idx1-ubyte

source("loadMNIST.R")
load_mnist() #load into the train list

labels = train$y;
labels[labels==0] <- 10; # Remap 0 to 10

images <- t(train$x)
#  scale
images <- images/255

inputData = images;

# clear env. 
rm(images)
rm(train)

# For debugging purposes, you may wish to reduce the size of the input data
# in order to speed up gradient checking. 
# Here, we create synthetic dataset using random data for testing

DEBUG = true; # Set DEBUG to true when debugging.
if (DEBUG) {
    inputSize = 8;
    inputData = matrix(rnorm(inputSize*100),inputSize,100);
    labels = sample(10, 100, replace=TRUE);
}

# Randomly initialise theta
theta = 0.005 * rnorm(numClasses * inputSize);

##======================================================================
## STEP 2: Implement softmaxCost
#
#  Implement softmaxCost in softmaxCost.m. 

library(Matrix)
library(pracma) 
source("softmaxCost.R")
CG <- softmaxCost(theta,numClasses, inputSize, lambda,
                    inputData, labels)
grad <- CG$grad
                                     
##======================================================================
## STEP 3: Gradient checking
#
#  As with any learning algorithm, you should always check that your
#  gradients are correct before learning the parameters.
# 
source("computeNumericalGradient.R")
if (DEBUG) {
    numGrad = computeNumericalGradient(theta, softmaxCost,
                                    numClasses,
                                    inputSize,
                                    lambda,
                                    inputData,
                                    labels)

    # Use this to visually compare the gradients side by side
    print(cbind(numGrad, grad)); 

    # Compare numerically computed gradients with those computed analytically
    diff = diff = norm(numGrad-grad, type="2")/norm(numGrad+grad, type="2");
    print(diff); 
    # The difference should be small. 
    # In our implementation, these values are usually less than 1e-7.

    # When your gradients are correct, congratulations!
}

##======================================================================
## STEP 4: Learning parameters
#
#  Once you have verified that your gradients are correct, 
#  you can start training your softmax regression code using softmaxTrain
#  (which uses optim()).

maxIter = 100;
source("softmaxTrain.R")
softmaxModel = softmaxTrain(inputSize, numClasses, lambda,
                            inputData, labels, maxIter);
                          
# Although we only use 100 iterations here to train a classifier for the 
# MNIST data set, in practice, training for more iterations is usually
# beneficial.

##======================================================================
## STEP 5: Testing
#
#  You should now test your model against the test images.
#  To do this, you will first need to write softmaxPredict
#  (in softmaxPredict.m), which should return predictions
#  given a softmax model and the input data.

labels = test$y;
labels[labels==0] <- 10; # Remap 0 to 10

images <- t(test$x)
#  scale
images <- images/255

inputData = images;
# clear env.
rm(images)
rm(test)

# You will have to implement softmaxPredict in softmaxPredict.m
source("softmaxPredict.R")
pred = softmaxPredict(softmaxModel, inputData);

acc = mean(labels == pred);
print(paste("Accuracy: ", acc * 100));

# Accuracy is the proportion of correctly classified images
# After 100 iterations, the results for our implementation were:
#
# Accuracy: 92.200#
#
# If your values are too low (accuracy less than 0.91), you should check 
# your code for errors, and make sure you are training on the 
# entire data set of 60000 28x28 training images 
# (unless you modified the loading code, this should be the case)
