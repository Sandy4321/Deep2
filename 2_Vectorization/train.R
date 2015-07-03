##======================================================================
## STEP 0: Here we provide the relevant parameters values that will
#  allow your sparse autoencoder to get good filters; you do not need to 
#  change the parameters below.

visibleSize = 28*28;   # number of input units 
hiddenSize = 196;     # number of hidden units 
sparsityParam = 0.01;   # desired average activation of the hidden units.
# (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
   #  in the lecture notes). 
lambda = 0.003;     # weight decay parameter       
beta = 3;            # weight of sparsity penalty term       

##======================================================================
## STEP 1: Load Images

source("loadMNIST.R")
load_mnist() #load into the train list
images <- t(train$x)

labels = train$y;
patches <- images[,1:10000]

# We are using display_network from the autoencoder code
source("display_network.R")
display_network(patches[,1:100]); # Show the first 100 images
print(labels(1:10));

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
source("computeNumericalGradient.R")

# Now we can use it to check your cost function and derivative calculations
# for the sparse autoencoder.  
source("sparseAutoencoderCostVec.R")
source("sparseAutoencoderGradVec.R")

# painfull slow !!!!
tstart <- Sys.time()
numgrad <- computeNumericalGradient(theta, sparseAutoencoderCostVec,
                                        visibleSize = visibleSize,
                                        hiddenSize = hiddenSize,
                                        lambda = lambda, 
                                        sparsityParam = sparsityParam,
                                        beta = beta,
                                        data = patches[,1:10])
tend <- Sys.time()
tend - tstart

# control
tstart <- Sys.time()
cost <- sparseAutoencoderCostVec(theta, visibleSize, hiddenSize, lambda, 
                            sparsityParam, beta, patches[,1:10]);
grad <- sparseAutoencoderGradVec(theta, visibleSize, hiddenSize, lambda, 
                                 sparsityParam, beta, patches[,1:10]);
tend <- Sys.time()
tend - tstart


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



# objective <- function(theta) {
#     C <- sparseAutoencoderCostVec(theta, visibleSize, hiddenSize, lambda, 
#                                 sparsityParam, beta, patches);
#     return(C)
# }
# 
# gradient <- function(theta) { 
#     G <- sparseAutoencoderGradVec(theta, visibleSize, hiddenSize, lambda, 
#                                 sparsityParam, beta, patches);
# 
#     return(G)
# }


# optimize with optim() function L-BFGS-B
# output <- optim(theta, objective, gradient, method="CG", 
#                 control = list(trace=1, maxit=100))
# 
# opttheta <- output$par

source("sparseAutoencoderCostVec.R")
source("sparseAutoencoderGradVec.R")

output <- optim(theta, 
                sparseAutoencoderCostVec, 
                sparseAutoencoderGradVec, 
                visibleSize = visibleSize,
                hiddenSize = hiddenSize,
                lambda = lambda, 
                sparsityParam = sparsityParam,
                beta = beta,
                data = patches,
                method="CG", 
                control = list(trace=1, maxit=100))

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






