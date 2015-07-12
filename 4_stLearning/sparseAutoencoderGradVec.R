sparseAutoencoderGradVec <- function(theta, visibleSize, hiddenSize, 
                                  lambda, sparsityParam, beta, data) {

# VECTORIZED sparseAutoencoder gradient calculation

# visibleSize: the number of input units (probably 64) 
# hiddenSize: the number of hidden units (probably 25) 
# lambda: weight decay parameter
# sparsityParam: The desired average activation for the hidden units (denoted in the lecture
#                           notes by the greek alphabet rho, which looks like a lower-case "p").
# beta: weight of sparsity penalty term
# data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
# The input theta is a vector (because minFunc expects the parameters to be a vector). 
# We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
# follows the notation convention of the lecture notes. 

W1 = matrix(theta[1:(hiddenSize*visibleSize)], hiddenSize, visibleSize);
W2 = matrix(theta[(hiddenSize*visibleSize+1):(2*hiddenSize*visibleSize)], visibleSize, hiddenSize);
b1 = theta[(2*hiddenSize*visibleSize+1):(2*hiddenSize*visibleSize+hiddenSize)];
b2 = theta[(2*hiddenSize*visibleSize+hiddenSize+1):length(theta)];

# Cost and gradient variables (your code needs to compute these values). 
# Here, we initialize them to zeros. 

W1grad = matrix(0,nrow(W1),ncol(W1)); 
W2grad = matrix(0,nrow(W2),ncol(W2));
b1grad = rep(0, length(b1)); 
b2grad = rep(0, length(b2));
rho_hat = rep(0, hiddenSize)

## ---------- YOUR CODE HERE --------------------------------------
#  Instructions: Compute the cost/optimization objective J_sparse(W,b) 
#  for the Sparse Autoencoder, and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
#
# W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
# Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
# as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
# respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
# with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
# [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
# of the lecture notes (and similarly for W2grad, b1grad, b2grad).
# 
# Stated differently, if we were using batch gradient descent to optimize the parameters,
# the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
# 
#  +(a) Implement forward propagation in your neural network, and implement the 
#      squared error term of the cost function.  Implement backpropagation to 
#      compute the derivatives.   Then (using lambda=beta=0), run Gradient Checking 
#      to verify that the calculations corresponding to the squared error cost 
#      term are correct.
#
#  -(b) Add in the weight decay term (in both the cost function and the derivative
#      calculations), then re-run Gradient Checking to verify correctness. 
#
#  -(c) Add in the sparsity penalty term, then re-run Gradient Checking to 
#      verify correctness.

# data <- patches for debug

m <- dim(data)[2]
# Part 1: Feedforward the neural network and return the cost

# W1 - 196x784
# W2 - 784x196
A1 <- data #784x10000
A2 <- sigmoid(W1%*%A1 + b1) #196x10000
A3 <- sigmoid(W2%*%A2 + b2) #784x10000

#add sparsity penalty
rho_hat <- apply(A2, 1, sum)
rho_hat <- rho_hat/m
rho <- sparsityParam
sparsity_delta = - rho/rho_hat + (1-rho)/(1-rho_hat)

# Part 2: Implement the backpropagation algorithm

delta3 <- -(A1-A3)*A3*(1-A3) # 784x10000
delta2 <-  (t(W2)%*%delta3 + beta*sparsity_delta)*A2*(1-A2) # 196x10000

W2grad = delta3%*%t(A2) #784x196
W1grad = delta2%*%t(A1) #196x784
b2grad = apply(delta3,1,sum)
b1grad = apply(delta2,1,sum)

W2grad = W2grad/m + W2*lambda
W1grad = W1grad/m + W1*lambda
b2grad = b2grad/m
b1grad = b1grad/m

#-------------------------------------------------------------------
# After computing the cost and gradient, we will convert the gradients back
# to a vector format (suitable for minFunc).  Specifically, we will unroll
# your gradient matrices into a vector.

    grad = c(as.vector(W1grad) , as.vector(W2grad), b1grad, b2grad);
    return(grad)

}    

#-------------------------------------------------------------------
# Here's an implementation of the sigmoid function, which you may find useful
# in your computation of the costs and the gradients.  This inputs a (row or
# column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

sigmoid <- function(x) {
  return(1 / (1 + exp(-x))) ### vectoraized
}

