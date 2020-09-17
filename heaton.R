library(Ryacas)

derivative = function(value, formula)
{
  x <- value
  return(eval(formula))
}

#activationFunction
#sigmoid
x <- "x"
e <- exp(1)
s <- expression(1 / (1+e^-x))
#derivative
formula <- D(s,x)		

delta = function(sum,E)
{
  #delta(score.sum,E)
  
  return(-1*(E * derivative(sum, formula)))
}

#heaton's example
x=1.13
(1/(1+exp(-x)))*(1-(1/(1+exp(-x))))


sigmoid = function(x){
  
  return(1 / (1+e^-x))
}


#c(input, hidden,output)
layers <- c(2,2,1)
#inputs
data <- c(1,0,1)
X <- data[1:2]
Y <- data[3]

D <- layers[1]
H <-  layers[2]

#matching page 42 of Heaton's book

#input <- matrix(c(1,0), nrow=2, ncol=1)
#W1 <- 0.01*matrix(rnorm(D*H), nrow=D, ncol=H)
#W1 <- matrix(0, nrow=head(layers,1), ncol=layers[2])
W1 <- matrix(c(-.07,.22,.94,.46), nrow=head(layers,1),ncol=layers[2])
#b1 <- matrix(0, nrow=1, ncol=H)
b1 <- matrix(c(-.46,.10), nrow=1, ncol=H)

#output
# number of categories for classification
#K <- length(unique(Y))
K <- tail(layers,1)

# create and init weights and bias 
#W2 <- 0.01*matrix(rnorm(H*K), nrow=H, ncol=K)
#W2 <- matrix(0, nrow=H, ncol=K)
W2 <- matrix(c(-.22,.58), nrow=H, ncol=K)
b2 <- matrix(.78, nrow=1, ncol=K)

#sigmoid (vs pmax)
#sum
hidden.layer.sum <- sweep(X %*% W1 ,2, b1, '+')
#output
hidden.layer.output <- sigmoid(hidden.layer.sum)

#threshold is either measured at end in output and class determination, or using Rectified Linear
#sum
score.sum <- sweep(hidden.layer.output %*% W2, 2, b2, '+')
#output
score.output <- sigmoid(score.sum)

#back propagation
E <- score.output - Y

#p44 
#interesting.  I'm doing a derivative of the sum (look at delta function, as if it's sigmoid, but sum != sigmoid algorithm)
#I guess whatever formula is derived from derivative, should be applied to the sum as well as output (partial derivatives)
#example.  If switching to tanh
score.output.delta <- delta(score.sum,E)

#node Delta

#derivative of sums
#p45
#derivative(1.05,formula) * (0.05*.58)
#only done on hidden layers

#derivative(hidden.layer.sum[2]) * score.output.delta * W2[2]
hidden.layer.nodeDeltas <- t(derivative(hidden.layer.sum, formula)) * (W2 %*% score.output.delta)

hidden.layer.gradient <- t(hidden.layer.output) %*% score.output.delta
