# dimension: 2X2
input <- matrix(1:4, nrow=2, ncol=2)
# dimension: 2x3
weights <- matrix(1:6, nrow=2, ncol=3)
# dimension: 1*3
bias <- matrix(1:3, nrow=1, ncol=3)
# doesn't work since unmatched dimension
#input %*% weights + bias

# solution 2: sweep addition
s2 <- sweep(input %*% weights ,2, bias, '+')

pmax(s2, 0)

iris

predict.dnn <- function(model, data = X.test) {
  # new data, transfer to matrix
  new.data <- data.matrix(data)
  
  # Feed Forwad
  hidden.layer <- sweep(new.data %*% model$W1 ,2, model$b1, '+')
  # neurons : Rectified Linear
  hidden.layer <- pmax(hidden.layer, 0)
  score <- sweep(hidden.layer %*% model$W2, 2, model$b2, '+')
  
  # Loss Function: softmax
  score.exp <- exp(score)
  probs <-sweep(score.exp, 1, rowSums(score.exp), '/') 
  
  # select max possiblity
  labels.predicted <- max.col(probs)
  return(labels.predicted)
}

# Train: build and train a 2-layers neural network 
train.dnn <- function(x, y, traindata=data, testdata=NULL,
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
                      display = 100,
                      random.seed = 1)
{
  #run this to init/test
  if(FALSE)
  {
    x=1:4
    y=5
    traindata=iris[samp,]
    testdata=iris[-samp,]
    hidden=6
    maxit=2000
    display=50
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
    random.seed = 1
  }
  # to make the case reproducible.
  set.seed(random.seed)
  
  # total number of training set
  N <- nrow(traindata)
  
  # extract the data and label
  # don't need atribute 
  X <- unname(data.matrix(traindata[,x]))
  Y <- traindata[,y]
  if(is.factor(Y)) { Y <- as.integer(Y) }
  # updated: 10.March.2016: create index for both row and col
  Y.len   <- length(unique(Y))
  Y.set   <- sort(unique(Y))
  Y.index <- cbind(1:N, match(Y, Y.set))
  
  # number of input features
  D <- ncol(X)
  # number of categories for classification
  K <- length(unique(Y))
  H <-  hidden
  
  # create and init weights and bias 
  W1 <- 0.01*matrix(rnorm(D*H), nrow=D, ncol=H)
  b1 <- matrix(0, nrow=1, ncol=H)
  
  W2 <- 0.01*matrix(rnorm(H*K), nrow=H, ncol=K)
  b2 <- matrix(0, nrow=1, ncol=K)
  
  # use all train data to update weights since it's a small dataset
  batchsize <- N
  # updated: March 17. 2016
  # init loss to a very big value
  loss <- 100000
  
  # Training the network
  i <- 0
  while(i < maxit && loss > abstol ) {
    
    # iteration index
    i <- i +1
    
    # forward ....
    # 1 indicate row, 2 indicate col
    hidden.layer <- sweep(X %*% W1 ,2, b1, '+')
    # neurons : ReLU
    hidden.layer <- pmax(hidden.layer, 0)
    score <- sweep(hidden.layer %*% W2, 2, b2, '+')
    
    # softmax
    score.exp <- exp(score)
    probs <-sweep(score.exp, 1, rowSums(score.exp), '/') 
    
    # compute the loss
    corect.logprobs <- -log(probs[Y.index])
    data.loss  <- sum(corect.logprobs)/batchsize
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
                       b1 = b1, 
                       W2 = W2, 
                       b2 = b2)
        labs <- predict.dnn(model, testdata[,-y])      
        # updated: 10.March.2016
        accuracy <- mean(as.integer(testdata[,y]) == Y.set[labs])
        cat(i, loss, accuracy, "\n")
      } else {
        cat(i, loss, "\n")
      }
    }
    
    # backward ....
    dscores <- probs
    dscores[Y.index] <- dscores[Y.index] -1
    dscores <- dscores / batchsize
    
    #weightDelta = output * OutputDelta
    dW2 <- t(hidden.layer) %*% dscores 
    db2 <- colSums(dscores)
    
    dhidden <- dscores %*% t(W2)
    dhidden[hidden.layer <= 0] <- 0
    
    #weightDelta <- output * HNodeDelta
    dW1 <- t(X) %*% dhidden
    
    #
    db1 <- colSums(dhidden) 
    
    # update ....
    
    #modify weightDelta w + current value of weight * regularization, ~momentum w current weight
    dW2 <- dW2 + reg*W2
    dW1 <- dW1  + reg*W1
    
    #apply learning rate to node delta and add weight
    W1 <- W1 - lr * dW1
    b1 <- b1 - lr * db1
    
    W2 <- W2 - lr * dW2
    b2 <- b2 - lr * db2
    
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
                 b1= b1, 
                 W2= W2, 
                 b2= b2)
  
  return(model)
}

########################################################################
# testing
#######################################################################
set.seed(1)

# 0. EDA
summary(iris)
plot(iris)

# 1. split data into test/train
samp <- c(sample(1:50,25), sample(51:100,25), sample(101:150,25))

# 2. train model
ir.model <- train.dnn(x=1:4, y=5, traindata=iris[samp,], testdata=iris[-samp,], hidden=6, maxit=2000, display=50)

# 3. prediction
labels.dnn <- predict.dnn(ir.model, iris[-samp, -5])

# 4. verify the results
table(iris[-samp,5], labels.dnn)
#          labels.dnn
#            1  2  3
#setosa     25  0  0
#versicolor  0 24  1
#virginica   0  0 25

#accuracy
mean(as.integer(iris[-samp, 5]) == labels.dnn)
# 0.98

library(Ryacas)

x <- "x"
s <- expression(1 / (1+e^-x))
#derivative
formula <- D(s,x)		
x <- 1.254
e <- exp(1)
eval(formula)


x=1.1254

#heaton's example
x=1.13
(1/(1+exp(-x)))*(1-(1/(1+exp(-x))))


