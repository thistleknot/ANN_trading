library(Ryacas)
library(parallel)
library(MLmetrics)

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

delta = function(output,E)
{
  #delta(score.sum,E)
  
  return(-1*(E * derivative(output, formula)))
}

#heaton's example
#x=1.13
#(1/(1+exp(-x)))*(1-(1/(1+exp(-x))))

#derivative(1.254,formula)
#(1-sigmoid(1.254))*sigmoid(1.254)

sigmoid = function(x)
{
  return(1 / (1+e^-x))
}

# Prediction
predict.dnn <- function(model, data = X.test) {
  # new data, transfer to matrix
  new.data <- data.matrix(data)
  
  # Feed Forwad
  hidden.layer <- sweep(new.data %*% model$W1 ,2, model$Wb1, '+')
  # neurons : Rectified Linear
  #hidden.layer <- pmax(hidden.layer, 0)
  #score <- sweep(hidden.layer %*% model$W2, 2, model$b2, '+')
  score.sum <- sweep(hidden.layer %*% model$W2, 2, model$Wb2, '+')
  
  # Loss Function: softmax
  #score.exp <- exp(score)
  #probs <-sweep(score.exp, 1, rowSums(score.exp), '/') 
  score.output <- sigmoid(score.sum)
  
  # select max possiblity
  #labels.predicted <- max.col(probs)
  labels.predicted <- score.output
  return(labels.predicted)
}

train.dnn <- function(x, y, traindata=data, testdata=NULL,
                      model = NULL,
                      # set hidden layers and neurons
                      # currently, only support 1 hidden layer
                      hidden=c(6), 
                      # max iteration steps
                      maxit=2000,
                      # delta loss 
                      abstol=1e-2,
                      # learning rate
                      lr = 1e-2,
                      # regularization rate
                      reg = 1e-3,
                      # show results every 'display' step
                      N = nrow(traindata),
                      display = 100,
                      #momentum
                      mom = .3,
                      random.seed = 1,
                      batchsize = N)
{
  
  if(FALSE)
  {
    #traindata=data
    #testdata=NULL
    model = NULL
    # set hidden layers and neurons
    # currently, only support 1 hidden layer
    hidden=c(6)
    # max iteration steps
    maxit=2000
    # delta loss 
    abstol=1e-2
    # learning rate
    lr = 1e-2
    # regularization rate
    reg = 1e-3
    # show results every 'display' step
    display = 100
    #momentum
    mom = .3
    random.seed = 1
    N = nrow(traindata)
    batchsize = N
    x=2:10
    y=1
    #traindata=data[samp,]
    #testdata=data[-samp,]
    hidden=6
    maxit=2000
    display=50
  }
  # to make the case reproducible.
  set.seed(random.seed)
  
  # total number of training set
  N <- nrow(traindata)
  
  #setup
  
  #c(input, hidden,output)
  #iteration
  
  #for classification, number of output neurons set to unique values (K)
  #http://www.parallelr.com/r-deep-neural-network-from-scratch/
  
  X <- data.matrix(traindata[,x])
  Y <- data.matrix(traindata[,y])
  
  if(is.null(model)) {
    # number of input features
    D <- ncol(X)
    # number of categories for classification
    K=1
    H <-  hidden
    
    layers <- c(D,H,K)
    #inputs
    #data <- c(1,0,1)
    #X <- data[1:2]
    #Y <- data[3]
    
    #matching page 42 of Heaton's book
    
    #input <- matrix(c(1,0), nrow=2, ncol=1)
    #W1 <- 0.01*matrix(rnorm(D*H), nrow=D, ncol=H)
    #W1 <- matrix(0, nrow=head(layers,1), ncol=layers[2])
    #W1 <- 0.01*matrix(rnorm(layers[1]*layers[2]), nrow=layers[1], ncol=layers[2])
    W1 <- matrix(rnorm(layers[1]*layers[2]), nrow=layers[1], ncol=layers[2])/sqrt(layers[1]*layers[2])
    #W1 <- matrix(c(-.07,.22,.94,.46), nrow=head(layers,1),ncol=layers[2])
    Wb1 <- matrix(0, nrow=1, ncol=layers[2])
    #Wb1 <- matrix(c(-.46,.10), nrow=1, ncol=layers[2])
    
    #output
    # number of categories for classification
    #K <- length(unique(Y))
    K <- tail(layers,1)
    
    # create and init weights and bias 
    #W2 <- 0.01*matrix(rnorm(layers[2]*layers[3]), nrow=layers[2], ncol=layers[3])
    W2 <- matrix(rnorm(layers[2]*layers[3]), nrow=layers[2], ncol=layers[3])/sqrt(layers[2]*layers[3])
    #W2 <- matrix(0, nrow=layers[2], ncol=layers[3])
    #W2 <- matrix(c(-.22,.58), nrow=H, ncol=K)
    #Wb2 <- matrix(.78, nrow=1, ncol=K)
    Wb2 <- matrix(0, nrow=1, ncol=layers[3])
    
    hidden.layer.lastWeightDelta = 0
    input.layer.lastWeightDelta = 0
    Wb2.layer.lastWeightDelta = 0
    Wb1.layer.lastWeightDelta = 0
  }
  
  loss <- 100000
  
  i <- 0
  while(i < maxit && loss > abstol ) {
    print(i)
    print(loss)
    
    # iteration index
    i <- i +1
    
    #sigmoid (vs pmax)
    #sum
    
    hidden.layer.sum <- sweep(X %*% W1 ,2, Wb1, '+')
    #output
    hidden.layer.output <- as.matrix(sigmoid(hidden.layer.sum))
    
    #threshold is either measured at end in output and class determination, or using Rectified Linear
    #sum
    score.sum <- sweep(hidden.layer.output %*% W2, 2, Wb2, '+')
    #output
    score.output <- sigmoid(score.sum)
    
    #back propagation
    #matt mazur /2 but "differntiates later" not sure what that means, so I removed its
    #E <- ((score.output - Y)^2)
    
    # compute the loss
    
    diff <- score.output - Y
    data.loss <- (t(diff) %*% diff) / (2 * N)
    reg.loss   <- 0.5*reg* (sum(W1*W1) + sum(W2*W2))
    loss <- data.loss + reg.loss
    
    # display results and update model
    if( i %% display == 0) {
      if(!is.null(testdata)) {
        model <- list( D = D,
                       H = H,
                       K = K,
                       # weights and bias
                       W1 = W1, 
                       Wb1 = Wb1, 
                       W2 = W2, 
                       Wb2 = Wb2)
        preds <- predict.dnn(model, testdata[,-y])      
        # updated: 10.March.2016
        accuracy <- MAPE(preds,testdata[,y])
        cat(i, loss, accuracy, "\n")
      } else {
        cat(i, loss, "\n")
      }
    }
    
    #p44 
    #interesting.  I'm doing a derivative of the sum (look at delta function, as if it's sigmoid, but sum != sigmoid algorithm)
    #I guess whatever formula is derived from derivative, should be applied to the sum as well as output (partial derivatives)
    #example.  If switching to tanh
    #heaton made an error.  Book states the derivative of the activation function, but he supplied the sum
    #two examples show the output is supposed to be used: 
    #https://github.com/thistleknot/Ann-v2/blob/master/myNueralNet.cpp
    #https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    
    #heaton's work
    #score.outputNode.delta <- delta(score.sum,E)
    
    score.outputNode.delta <- data.matrix(delta(score.sum,diff))
    
    #node Delta
    
    #derivative of sums
    #p45
    #derivative(1.05,formula) * (0.05*.58)
    #only done on hidden layers
    
    #derivative(hidden.layer.sum[2]) * score.outputNode.delta * W2[2]
    #no need to calculate nodeDelta's for input or bias nuerons (#43)
    #according to https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
    #error should be 1/2(E^2), but also derivative is based on output vs sum
    #p46 states, "sum is for derivatives (node deltas), output is for gradient)
    
    #heaton's error
    #hidden.layerNode.delta <- t(derivative(hidden.layer.sum, formula)) * (W2 %*% score.outputNode.delta)
    #matt mazur
    
    hidden.layerNode.delta <-  t(derivative(hidden.layer.sum, formula)) %*% score.outputNode.delta %*% t(W2)
    
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
    #input.layer.gradient <- t(sigmoid(X)) %*% hidden.layerNode.delta
    #p45
    #"multiply the node delta for the target neuron"
    input.layer.gradient <- X %*% data.matrix(1/W1*W1) %*% hidden.layerNode.delta 
    
    #I don't seem to be using W1
    #that's because W1 is adjusted based on incoming back propagated values (similar to how the output neuron is calculated)
    #hence why we do the derivative of the sigmoid when we use the output, it's like the inverse
    
    #db2 <- colSums(dscores)
    #bias neurons outputs are 1
    Wb2.layer.gradient <- sigmoid(1) * score.outputNode.delta
    Wb1.layer.gradient <- t(sigmoid(1) * hidden.layerNode.delta)
    
    # update ....
    #would normally batch sum gradients per set of input elements
    #of if doing rprop, similar but compare signs
    
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
    W2 <- W2 + (hidden.layer.WeightDelta)
    W1 <- W1 + colSums(input.layer.WeightDelta)
    Wb2 <- Wb2 + colSums(Wb2.layer.WeightDelta)
    Wb1 <- Wb1 + colSums(Wb1.layer.WeightDelta)
  }
  
  # final results
  # creat list to store learned parameters
  # you can add more parameters for debug and visualization
  # such as residuals, fitted.values ...
  model <- list( D = D,
                 H = H,
                 K = K,
                 # weights and bias
                 W1= W1, 
                 Wb1= Wb1, 
                 W2= W2, 
                 Wb2= Wb2)
  
  return(model)
}

#testing
set.seed(1)

data <- read.csv(file="states.csv", header=T)[,-1]

samp <- c(sample(1:nrow(data),nrow(data)*.8))

traindata=data[samp,]
testdata=data[-samp,]

trainParam <- caret::preProcess(as.matrix(traindata))
traindata <- predict(trainParam,traindata)
testdata <- predict(trainParam,testdata)

train.model <- train.dnn(x=2:10, y=1, traindata, testdata, hidden=6, maxit=2000, display=50)
test.predictions <- predict.dnn(train.model, data[-samp,-1])

accuracy <- MAPE(unlist(test.predictions),testdata[,1])

