feedForwardAutoencoder <- function(theta, hiddenSize, visibleSize, data) {

# theta: trained weights from the autoencoder
# visibleSize: the number of input units (probably 64) 
# hiddenSize: the number of hidden units (probably 25) 
# data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
  
# We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
# follows the notation convention of the lecture notes. 

W1 = matrix(theta[1:(hiddenSize*visibleSize)], hiddenSize, visibleSize);
b1 = theta[(2*hiddenSize*visibleSize+1):(2*hiddenSize*visibleSize+hiddenSize)];

## ---------- YOUR CODE HERE --------------------------------------
#  Instructions: Compute the activation of the hidden layer for the Sparse Autoencoder.

A1 <- data #784x15000
A2 <- sigmoid(W1%*%A1 + b1) #200x15000
return(A2)

#-------------------------------------------------------------------

}

#-------------------------------------------------------------------
# Here's an implementation of the sigmoid function, which you may find useful
# in your computation of the costs and the gradients.  This inputs a (row or
# column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

sigmoid <- function(x) {
    return(1 / (1 + exp(-x))) ### vectoraized
}
