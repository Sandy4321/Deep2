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


for (i in 1:m) {
    basis = rep(0,m)
    basis[i]=1
    print(i)
    #if(m %% 1 == 0) print(paste(m, "theta processed"))
    numgrad[i] = (J(theta+EPSILON*basis,...) - J(theta-EPSILON*basis,...))/(2*EPSILON)
}


return(numgrad)
## ---------------------------------------------------------------
}
