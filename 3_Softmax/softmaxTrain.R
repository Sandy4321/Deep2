softmaxTrain <- function(inputSize, numClasses, lambda, inputData, labels, maxIter) {
#softmaxTrain Train a softmax model with the given parameters on the given
# data. Returns softmaxOptTheta, a vector containing the trained parameters
# for the model.
#
# inputSize: the size of an input vector x^(i)
# numClasses: the number of classes 
# lambda: weight decay parameter
# inputData: an N by M matrix containing the input data, such that
#            inputData(:, c) is the cth input
# labels: M by 1 matrix containing the class labels for the
#            corresponding inputs. labels(c) is the class label for
#            the cth input
#
# maxIter: number of iterations to train for

# initialize parameters
theta = 0.005 * rnorm(numClasses * inputSize);

# Use optim() to minimize the function
objective <- function(theta) {
    cost <- softmaxCost(theta,numClasses, inputSize, lambda,
                inputData, labels)$cost
    return(cost)
}

gradient <- function(theta) { 
    grad <- softmaxCost(theta,numClasses, inputSize, lambda,
                        inputData, labels)$grad    
    return(grad)
}


# optimize with optim() function L-BFGS-B
source("softmaxCost.R")

optdata <- optim(theta, 
                objective,
                gradient,
                method="CG", 
                control = list(trace=1, maxit=maxIter))



# Fold softmaxOptTheta into a nicer format
softmaxModel <- list(optTheta = matrix(optdata$par, numClasses, inputSize),
                    inputSize = inputSize,
                    numClasses = numClasses,
                    cost = optdata$value)
                          
}                      
