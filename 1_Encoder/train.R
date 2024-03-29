## CS294A/CS294W Programming Assignment Starter Code

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  programming assignment. You will need to complete the code in sampleIMAGES.m,
#  sparseAutoencoderCost.m and computeNumericalGradient.m. 
#  For the purpose of completing the assignment, you do not need to
#  change the code in this file. 
#
##======================================================================
## STEP 0: Here we provide the relevant parameters values that will
#  allow your sparse autoencoder to get good filters; you do not need to 
#  change the parameters below.

visibleSize = 8*8;   # number of input units 
hiddenSize = 25;     # number of hidden units 
sparsityParam = 0.01;   # desired average activation of the hidden units.
# (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
   #  in the lecture notes). 
lambda = 0.0001;     # weight decay parameter       
beta = 3;            # weight of sparsity penalty term       

##======================================================================
## STEP 1: Implement sampleIMAGES
#
#  After implementing sampleIMAGES, the display_network command should
#  display a random sample of 200 patches from the dataset

library(R.matlab)
source("sampleIMAGES.R")
patches <- sampleIMAGES();
source("display_network.R")
patches_index <- sample(1:ncol(patches),200, replace=FALSE)
display_network(patches[,patches_index]);
display_network(patches[,1:200])


#  Obtain random parameters theta
source("initializeParameters.R")
theta = initializeParameters(hiddenSize, visibleSize);

##======================================================================
## STEP 2: Implement sparseAutoencoderCost
#
#  You can implement all of the components (squared error cost, weight decay term,
#  sparsity penalty) in the cost function at once, but it may be easier to do 
#  it step-by-step and run gradient checking (see STEP 3) after each step.  We 
#  suggest implementing the sparseAutoencoderCost function using the following steps:
#

#  Feel free to change the training settings when debugging your
#  code.  (For example, reducing the training set size or 
#  number of hidden units may make your code run faster; and setting beta 
#  and/or lambda to zero may be helpful for debugging.)  However, in your 
#  final submission of the visualized weights, please use parameters we 
#  gave in Step 0 above.

##======================================================================
    ## STEP 3: Gradient Checking
#
# Hint: If you are debugging your code, performing gradient checking on smaller models 
# and smaller training sets (e.g., using only 10 training examples and 1-2 hidden 
                             # units) may speed things up.

# First, lets make sure your numerical gradient computation is correct for a
# simple function.  After you have implemented computeNumericalGradient.m,
# run the following: 
source("checkNumericalGradient.R")
source("computeNumericalGradient.R")
checkNumericalGradient();

# Now we can use it to check your cost function and derivative calculations
# for the sparse autoencoder.  

source("sparseAutoencoderCost.R")

numgrad <- computeNumericalGradient(theta, sparseAutoencoderCost,
                                        visibleSize = visibleSize,
                                        hiddenSize = hiddenSize,
                                        lambda = lambda, 
                                        sparsityParam = sparsityParam,
                                        beta = beta,
                                        data = patches[,1:10])

CG <- sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, 
                            sparsityParam, beta, patches[,1:10]);
cost <- CG$cost
grad <- CG$grad


# Use this to visually compare the gradients side by side
print(cbind(numgrad, grad)); 

# Compare numerically computed gradients with the ones obtained from backpropagation
diff = norm(numgrad-grad, type="2")/norm(numgrad+grad, type="2");
print(diff); # Should be small. In our implementation, these values are
# usually less than 1e-9.

# When you got this working, Congratulations!!!

##======================================================================
## STEP 4: After verifying that your implementation of
#  sparseAutoencoderCost is correct, You can start training your sparse
#  autoencoder with minFunc (L-BFGS).

#  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, visibleSize);

# Here, we use L-BFGS to optimize our cost
# function. Generally, for minFunc to work, you
# need a function pointer with two outputs: the
# function value and the gradient. In our problem,
# sparseAutoencoderCost.m satisfies this.

max_itertions = 400;      # Maximum number of iterations of L-BFGS to run 
invisible = 0;              # Display on
theta = initializeParameters(hiddenSize, visibleSize);


objective <- function(theta) {
    CG <- sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, 
                                sparsityParam, beta, patches);
    return(CG$cost)
}

source("sparseAutoencoderCostLight.R") # wihtout gradient calculation
objective_light <- function(theta) {
    cost <- sparseAutoencoderCostLight(theta, visibleSize, hiddenSize, lambda, 
                                       sparsityParam, beta, patches);
    return(cost)
}

gradient <- function(theta) { 
    CG <- sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, 
                                sparsityParam, beta, patches);

    return(CG$grad)
}


# optimize with optim() function
output <- optim(theta, objective_light, gradient, method="L-BFGS-B", 
                control = list(trace=1, maxit=1000))

opttheta <- output$par





##======================================================================
## STEP 5: Visualization 
# optim result
W1 = matrix(opttheta[1:(hiddenSize*visibleSize)], hiddenSize, visibleSize);
W1<-t(W1)
display_network(W1); 


# reserved plot for control display_network
# par(mfrow=c(5,5),
#     xaxt="n",
#     yaxt="n",
#     mai=c(0.02,0.02,0.02,0.02))
# a=1
# image(1:8,1:8,matrix(W1[,a],8,8))
# a=a+1
# image(1:8,1:8,matrix(W1[,a],8,8))
# a=a+1
# image(1:8,1:8,matrix(W1[,a],8,8))
# a=a+1
# image(1:8,1:8,matrix(W1[,a],8,8))
# a=a+1
# image(1:8,1:8,matrix(W1[,a],8,8))
# a=a+1
# image(1:8,1:8,matrix(W1[,a],8,8))
# a=a+1
# image(1:8,1:8,matrix(W1[,a],8,8))
# a=a+1
# image(1:8,1:8,matrix(W1[,a],8,8))
# a=a+1
# image(1:8,1:8,matrix(W1[,a],8,8))
# a=a+1
# image(1:8,1:8,matrix(W1[,a],8,8))
# a=a+1
# image(1:8,1:8,matrix(W1[,a],8,8))
# a=a+1
# image(1:8,1:8,matrix(W1[,a],8,8))
# a=a+1
# image(1:8,1:8,matrix(W1[,a],8,8))
# a=a+1
# image(1:8,1:8,matrix(W1[,a],8,8))
# a=a+1
# image(1:8,1:8,matrix(W1[,a],8,8))
# a=a+1
# image(1:8,1:8,matrix(W1[,a],8,8))
# a=a+1
# image(1:8,1:8,matrix(W1[,a],8,8))
# a=a+1
# image(1:8,1:8,matrix(W1[,a],8,8))
# a=a+1
# image(1:8,1:8,matrix(W1[,a],8,8))
# a=a+1
# image(1:8,1:8,matrix(W1[,a],8,8))
# a=a+1
# image(1:8,1:8,matrix(W1[,a],8,8))
# a=a+1
# image(1:8,1:8,matrix(W1[,a],8,8))
# a=a+1
# image(1:8,1:8,matrix(W1[,a],8,8))
# a=a+1
# image(1:8,1:8,matrix(W1[,a],8,8))
# a=a+1
# image(1:8,1:8,matrix(W1[,a],8,8))






