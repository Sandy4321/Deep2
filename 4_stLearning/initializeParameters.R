initializeParameters <- function(hiddenSize, visibleSize) {

## Initialize parameters randomly based on layer sizes.
r  = sqrt(6) / sqrt(hiddenSize+visibleSize+1);   # we'll choose weights uniformly from the interval [-r, r]
W1 = matrix(runif(hiddenSize*visibleSize, min=-r, max=r), hiddenSize, visibleSize)
W2 = matrix(runif(hiddenSize*visibleSize, min=-r, max=r), visibleSize, hiddenSize)

b1 = matrix(0,hiddenSize, 1);
b2 = matrix(0,visibleSize, 1);

# Convert weights and bias gradients to the vector form.
# This step will "unroll" (flatten and concatenate together) all 
# your parameters into a vector, which can then be used with minFunc. 
theta = c(as.vector(W1), as.vector(W2), as.vector(b1), as.vector(b2));
}

