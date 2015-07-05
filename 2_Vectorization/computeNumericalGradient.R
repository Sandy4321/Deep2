computeNumericalGradient <- function(theta, J,...) {
# numgrad = computeNumericalGradient(J, theta)
# theta: a vector of parameters
# J: a function that outputs a real-number. Calling y = J(theta) will return the
# function value at theta. 
  
# Initialize numgrad with zeros
numgrad = rep(0,length(theta));
EPSILON = 0.0001

## ---------- YOUR CODE HERE --------------------------------------
# Instructions: 
# Implement numerical gradient checking, and return the result in numgrad.  
# (See Section 2.3 of the lecture notes.)
# You should write code so that numgrad(i) is (the numerical approximation to) the 
# partial derivative of J with respect to the i-th input argument, evaluated at theta.  
# I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
# respect to theta(i).
#                
# Hint: You will probably want to compute the elements of numgrad one at a time. 

m = length(theta)

# library(doSNOW)
# library(foreach)
# 
# cl<-makeCluster(4) #change the 2 to your number of CPU cores
# registerDoSNOW(cl)

# foreach(i=1:m) %dopar% {
#     basis = rep(0,m)
#     basis[i]=1
#     #print(i)
#     if(i %% 1000 == 0) print(i)
#     numgrad[i] = (J(theta+EPSILON*basis,...) - J(theta-EPSILON*basis,...))/(2*EPSILON)
# } 

# stopCluster(cl)

for (i in 1:m) {
    theta_minus <- theta
    theta_minus[i] <- theta_minus[i]-EPSILON
    theta_plus <- theta
    theta_plus[i] <- theta_plus[i]+EPSILON
    
    numgrad[i] = (J(theta_plus,...) - J(theta_minus,...))/(2*EPSILON)
    # debug print
    if(i %% 1000 == 0) {
        print(paste("theta #",i,"grad:",numgrad[i]))
    }    
}


return(numgrad)
## ---------------------------------------------------------------
}
