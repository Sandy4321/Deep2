## CS294A/CS294W Self-taught Learning Exercise

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  self-taught learning. You will need to complete code in feedForwardAutoencoder.m
#  You will also need to have implemented sparseAutoencoderCost.m and 
#  softmaxCost.m from previous exercises.
#
## ======================================================================
#  STEP 0: Here we provide the relevant parameters values that will
#  allow your sparse autoencoder to get good filters; you do not need to 
#  change the parameters below.

inputSize  = 28 * 28;
numLabels  = 5;
hiddenSize = 200;
sparsityParam = 0.1; # desired average activation of the hidden units.
                     # (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		             #  in the lecture notes). 
lambda = 3e-3;       # weight decay parameter       
beta = 3;            # weight of sparsity penalty term   
maxIter = 400;

## ======================================================================
#  STEP 1: Load data from the MNIST database
#
#  This loads our training and test data from the MNIST database files.
#  We have sorted the data for you in this so that you will not have to
#  change it.

# Load MNIST database files
source("loadMNIST.R")
load_mnist() #load into the train list

mnistLabels = train$y;
#labels[labels==0] <- 10; # Remap 0 to 10
mnistData = t(train$x)/255
rm(train)
# Set Unlabeled Set (All Images)
#unlabeledTrainingImages <- mnistData

# Simulate a Labeled and Unlabeled set
unlabeledSet = which(mnistLabels >= 5);
unlabeledData = mnistData[, unlabeledSet];


labeledSet   = which(mnistLabels >= 0 & mnistLabels <= 4);


numTrain = round(length(labeledSet)/2);
trainSet = labeledSet[1:numTrain];
testSet  = labeledSet[(numTrain+1):length(labeledSet)];



trainData   = mnistData[, trainSet];
trainLabels = mnistLabels[trainSet] + 1; # Shift Labels to the Range 1-5

testData   = mnistData[, testSet];
testLabels = mnistLabels[testSet] + 1;   # Shift Labels to the Range 1-5

# Output Some Statistics
print(paste("# examples in unlabeled set: ", dim(unlabeledData)[2]));
print(paste("# examples in supervised training set: ", dim(trainData)[2]));
print(paste("# examples in supervised testing set: ", dim(testData)[2]));


## ======================================================================
#  STEP 2: Train the sparse autoencoder
#  This trains the sparse autoencoder on the unlabeled training
#  images. 

#  Randomly initialize the parameters
source("initializeParameters.R")
theta = initializeParameters(hiddenSize, inputSize);

## ----------------- YOUR CODE HERE ----------------------
#  Find opttheta by running the sparse autoencoder on
#  unlabeledTrainingImages

objective <- function(theta) {
    cost <- sparseAutoencoderCostVec(theta, inputSize, hiddenSize, lambda, 
                                     sparsityParam, beta, unlabeledData);
    return(cost)
}
gradient <- function(theta) { 
    grad <- sparseAutoencoderGradVec(theta, inputSize, hiddenSize, lambda, 
                                     sparsityParam, beta, unlabeledData);
    return(grad)
}

# optimize with optim() function L-BFGS-B
source("sparseAutoencoderCostVec.R")
source("sparseAutoencoderGradVec.R")

tstart <- Sys.time()
output <- optim(theta, 
                objective,
                gradient,
                method="L-BFGS-B", 
                control = list(trace=1, maxit=maxIter))
opttheta <- output$par
tend <- Sys.time()
tend - tstart

save(opttheta, file="opttheta.RData")

## -----------------------------------------------------
                          
# Visualize weights
source("display_network.R")
W1 = matrix(opttheta[1:(hiddenSize*inputSize)], hiddenSize, inputSize);
W1<-t(W1)
display_network(W1[,1:100]);

##======================================================================
## STEP 3: Extract Features from the Supervised Dataset
#  
#  You need to complete the code in feedForwardAutoencoder.m so that the 
#  following command will extract features from the data.

source("feedForwardAutoencoder.R")
trainFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, 
                                       trainData);

testFeatures = feedForwardAutoencoder(opttheta, hiddenSize, inputSize, 
                                      testData);

##======================================================================
## STEP 4: Train the softmax classifier

 
## ----------------- YOUR CODE HERE ----------------------
#  Use softmaxTrain.R from the previous exercise to train a multi-class
#  classifier. 

#  Use lambda = 1e-4 for the weight regularization for softmax

# You need to compute softmaxModel using softmaxTrain on trainFeatures and
# trainLabels

library(Matrix)
library(pracma) 

lambda = 1e-4
maxIter = 100
numClasses = length(unique(trainLabels))

source("softmaxTrain.R")
source("softmaxCost.R")
softmaxModel = softmaxTrain(inputSize, numClasses, lambda, trainFeatures, trainLabels, maxIter);



## -----------------------------------------------------


##======================================================================
## STEP 5: Testing 

## ----------------- YOUR CODE HERE ----------------------
# Compute Predictions on the test set (testFeatures) using softmaxPredict
# and softmaxModel

source("softmaxPredict.R")
pred = softmaxPredict(softmaxModel, testFeatures);

acc = mean(testLabels == pred);
print(paste("Accuracy: ", acc * 100));













## -----------------------------------------------------

# Classification Score
fprintf('Test Accuracy: #f##\n', 100*mean(pred(:) == testLabels(:)));

# (note that we shift the labels by 1, so that digit 0 now corresponds to
#  label 1)
#
# Accuracy is the proportion of correctly classified images
# The results for our implementation was:
#
# Accuracy: 98.3#
#
# 
