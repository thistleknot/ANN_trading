library(Ryacas)
library(parallel)

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

#derivative(1.254,formula)
#(1-sigmoid(1.254))*sigmoid(1.254)

sigmoid = function(x)
{
  return(1 / (1+e^-x))
}

#constants/parameter's
# delta loss 
abstol=1e-2
# learning rate
lr = 1e-2
# regularization rate
reg = 1e-3
#momentum
mom = .3

#setup

#c(input, hidden,output)
#iteration
iter=0
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
Wb1 <- matrix(c(-.46,.10), nrow=1, ncol=H)

#output
# number of categories for classification
#K <- length(unique(Y))
K <- tail(layers,1)

# create and init weights and bias 
#W2 <- 0.01*matrix(rnorm(H*K), nrow=H, ncol=K)
#W2 <- matrix(0, nrow=H, ncol=K)
W2 <- matrix(c(-.22,.58), nrow=H, ncol=K)
Wb2 <- matrix(.78, nrow=1, ncol=K)

#sigmoid (vs pmax)
#sum
hidden.layer.sum <- sweep(X %*% W1 ,2, Wb1, '+')
#output
hidden.layer.output <- sigmoid(hidden.layer.sum)

#threshold is either measured at end in output and class determination, or using Rectified Linear
#sum
score.sum <- sweep(hidden.layer.output %*% W2, 2, Wb2, '+')
#output
score.output <- sigmoid(score.sum)

#back propagation
E <- score.output - Y

#p44 
#interesting.  I'm doing a derivative of the sum (look at delta function, as if it's sigmoid, but sum != sigmoid algorithm)
#I guess whatever formula is derived from derivative, should be applied to the sum as well as output (partial derivatives)
#example.  If switching to tanh
score.outputNode.delta <- delta(score.sum,E)

#node Delta

#derivative of sums
#p45
#derivative(1.05,formula) * (0.05*.58)
#only done on hidden layers

#derivative(hidden.layer.sum[2]) * score.outputNode.delta * W2[2]
#no need to calculate nodeDelta's for input or bias nuerons (#43)
hidden.layerNode.delta <- t(derivative(hidden.layer.sum, formula)) * (W2 %*% score.outputNode.delta)

#gradient works with outputs
#partial derivatives
#p45
#used in batch and summed before delta's are calculated

#output * delta
#skips weights
#dW2 <- t(hidden.layer) %*% dscores 

hidden.layer.gradient <- t(hidden.layer.output) %*% score.outputNode.delta
#dW1 <- t(X) %*% dhidden

#p45 to make sense of this.  Had to write out each InputOutput neuron to Hidden layer neuron's delta
input.layer.gradient <- t(do.call(cbind,mclapply(1:length(X), function(x)
  {
  return(X[x] * hidden.layerNode.delta)
  })))


#db2 <- colSums(dscores)
#bias neurons outputs are 1
Wb2.layer.gradient <- 1 %*% score.outputNode.delta
Wb1.layer.gradient <- t(1 * hidden.layerNode.delta)

# update ....
#would normally batch sum gradients per set of input elements
#of if doing rprop, similar but compare signs

if (iter==0)
{
  hidden.layer.lastWeightDelta = 0
  input.layer.lastWeightDelta = 0
  Wb2.layer.lastWeightDelta = 0
  Wb1.layer.lastWeightDelta = 0
}

hidden.layer.WeightDelta <- (lr * hidden.layer.gradient) + (mom * hidden.layer.lastWeightDelta)
input.layer.WeightDelta <- (lr * input.layer.gradient) + (mom * input.layer.lastWeightDelta)
Wb2.layer.WeightDelta <- (lr * Wb2.layer.gradient) + (mom * Wb2.layer.lastWeightDelta)
Wb1.layer.WeightDelta <- (lr * Wb1.layer.gradient) + (mom * Wb1.layer.lastWeightDelta)

hidden.layer.lastWeightDelta = hidden.layer.WeightDelta
input.layer.lastWeightDelta = input.layer.WeightDelta
Wb2.layer.lastWeightDelta = Wb2.layer.WeightDelta
Wb1.layer.lastWeightDelta = Wb1.layer.WeightDelta

#heaton doesn't use regularization

#dW2 <- dW2 + reg*W2
#dW1 <- dW1  + reg*W1
#W1 <- W1 - lr * dW1
#b1 <- b1 - lr * db1
#W2 <- W2 - lr * dW2
#b2 <- b2 - lr * db2

#"delta value will be added to the weight to alter the neural network for the first training iteration.
W2 <- W2 + hidden.layer.WeightDelta
W1 <- W1 + input.layer.WeightDelta
Wb2 <- Wb2 + Wb2.layer.WeightDelta
Wb1 <- Wb1 + Wb1.layer.WeightDelta

