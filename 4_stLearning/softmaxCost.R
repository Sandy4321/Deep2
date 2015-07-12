softmaxCost <- function(theta, numClasses, inputSize, lambda, data, labels) {

# numClasses - the number of classes 
# inputSize - the size N of the input vector
# lambda - weight decay parameter
# data - the N x M input matrix, where each column data(:, i) corresponds to
#        a single test set
# labels - an M x 1 matrix containing the labels corresponding for the input data
#

# Unroll the parameters from theta
theta = matrix(theta, numClasses, inputSize); #10x784

numCases = dim(data)[2]; #60.000

groundTruth = matrix(as.numeric(sparseMatrix(labels, 1:numCases)),
                     numClasses,numCases) #10x60.000

cost = 0;

thetagrad = matrix(0, numClasses, inputSize); #10x784

## ---------- YOUR CODE HERE --------------------------------------
#  Instructions: Compute the cost and gradient for softmax regression.
#                You need to compute thetagrad and cost.
#                The groundTruth matrix might come in handy.

# cost calculation
z = theta %*% data; # 10x784*784x60.000=10x60.000 

# preventing overflow
max_col <- apply(z,2,max) #60.000 max elements by the columns
max_col <- repmat(max_col,10,1) #minus max in each col
z <- z - max_col

h = exp(z); #10x60.000
sum_col <- repmat(apply(h,2,sum),10,1)
h <- h/sum_col
cost = (-1/numCases)*sum(groundTruth*log(h)) + (lambda/2)*sum(theta^2);

# grad calculation
thetagrad = (groundTruth - h)%*%t(data)
thetagrad = (-1/numCases)*thetagrad + lambda*theta 



# ------------------------------------------------------------------
# Unroll the gradient matrices into a vector for minFunc
return(list(cost=cost, grad=as.vector(thetagrad)))

}

