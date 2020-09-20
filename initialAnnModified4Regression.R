# Copyright 2016: www.ParallelR.com
# Parallel Blog : R For Deep Learning (I): Build Fully Connected Neural Network From Scratch
# Regression by 2-layers DNN and tested by iris dataset
# Description: Build 2-layers DNN to predict Petal.Width according to the other three variables in iris dataset.
# Author: Matrix Chen, xinyulux@gmail.com
# Modified on the basis of the original written by Peng Zhao.

# sigmoid
sigmoid <- function(z) {
  g <- 1 / (1 + exp(-1 * z))
  g
}


#modified from: https://github.com/thistleknot/Ann-v2/blob/master/myNueralNet.cpp

predict.dnn <- function(model, data = X.test) {
  # new data, transfer to matrix
  new.data <- data.matrix(data)
  
  # Feed Forwad
  hidden.layer <- sweep(new.data %*% model$W1 ,2, model$b1, '+')
  # neurons : Rectified Linear
  hidden.layer <- sigmoid(hidden.layer)
  #hidden.layer <- pmax(hidden.layer, 0)
  score <- sweep(hidden.layer %*% model$W2, 2, model$b2, '+')
  
  # Loss Function: softmax
  #score.exp <- exp(score)
  #probs <-sweep(score.exp, 1, rowSums(score.exp), '/') 
  
  # select max possiblity
  #labels.predicted <- max.col(probs)
  #return(labels.predicted)
  return(score)
  
}

# Train: build and train a 2-layers neural network 
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
                      display = 100,
                      random.seed = 1)
{
  #run this to init/test
  if(FALSE)
  {
    x=2:10
    y=1
    #traindata=data[samp,]
    #testdata=data[-samp,]
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
  #if(is.factor(Y)) { Y <- as.integer(Y) }
  # updated: 10.March.2016: create index for both row and col
  #Y.len   <- length(unique(Y))
  #Y.set   <- sort(unique(Y))
  #Y.index <- cbind(1:N, match(Y, Y.set))
  
  # create model or get model from parameter
  if(is.null(model)) {
    # number of input features
    D <- ncol(X)
    # only one output node for regression
    K <- 1
    H <-  hidden
    
    # create and init weights and bias 
    W1 <- matrix(rnorm(D*H), nrow=D, ncol=H)/sqrt(D*H)
    b1 <- matrix(0, nrow=1, ncol=H)
    
    W2 <- matrix(rnorm(H*K), nrow=H, ncol=K)/sqrt(H*K)
    b2 <- matrix(0, nrow=1, ncol=K)
  } else {
    D  <- model$D
    K  <- model$K
    H  <- model$H
    W1 <- model$W1
    b1 <- model$b1
    W2 <- model$W2
    b2 <- model$b2
  }
  
  
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
    #hidden.layer <- pmax(hidden.layer, 0)
    hidden.layer <- sigmoid(hidden.layer)
    score <- sweep(hidden.layer %*% W2, 2, b2, '+')
    
    diff <- score - Y
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
                       b1 = b1, 
                       W2 = W2, 
                       b2 = b2)
        labs <- predict.dnn(model, testdata[,-y])      
        # updated: 10.March.2016
        diff <- labs - testdata[,y]
        mse <- (t(diff) %*% diff) / (2 * N)
        cat(i, loss, mse, "\n")
        } else {
        cat(i, loss, "\n")
      }
    }
    
    # backward ....
    #dscores <- probs
    dscores <- score - Y
    #dscores[Y.index] <- dscores[Y.index] -1
    #dscores = delta
    
    #output Delta
    #dscores is Error
    dscores <- dscores / batchsize
    
    #should be adding gradients
    
    #error * output is what? (closest to node delta, but only after multiplied by weight)
    dW2 <- t(hidden.layer) %*% dscores 
    #why is it adding derivatives to the bias delta?
    db2 <- colSums(dscores)
    
    #output node delta * weight is prep step for node delta, to finish it off,
    #dhidden needs to be multiplied by derivative of hidden (done in dW1 & db1)
    dhidden <- dscores %*% t(W2)
    #dhidden[hidden.layer <= 0] <- 0
    
    #InputWeightGradient <- Ioutput * HNodeDelta
    #dW1 <- t(X) %*% dhidden
    
    #
    #db1 <- colSums(dhidden) 
    #****
    #*hiddenlayer * (1-hidden.layer) is the derivative of the sigmoid function
    #*#p44
    
    dW1 <- dhidden * (hidden.layer * (1-hidden.layer))
    dW1 <- t(X) %*% dW1
    
    #*db1 is node Delta: (da of H2 out * o1 node delta * weight of hidden to output)
    #*to become gradient, db1 needs to be multiplied by output of hidden (which is done in dW2, so it would be done twice)
    #*something isn't right with heaton's math, which makes me think this is why he was using sums
    db1 <- dhidden * (hidden.layer * (1-hidden.layer))
    db1 <- colSums(db1)
    #****
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

data <- read.csv(file="states.csv", header=T)[,-1]

# 0. EDA
summary(data)
plot(data)

# 1. split data into test/train

samp <- c(sample(1:nrow(data),nrow(data)*.8))

traindata=data[samp,]
testdata=data[-samp,]

trainParam <- caret::preProcess(as.matrix(traindata))
traindata <- predict(trainParam,traindata)
testdata <- predict(trainParam,testdata)

model <- train.dnn(x=2:10, y=1, traindata, testdata, hidden=6, maxit=2000, display=50)

test.predictions <- predict.dnn(model, data[-samp,-1])

plot(traindata)

test <- testdata
m <- length(test[,4])
diff <- test.predictions - test[,1]
mse <- (t(diff) %*% diff) / (2 * m)
mse

if(FALSE)
{
  # Visualization
  # the output from screen, copy and paste here.
  data1 <- ("i loss mse
        50 0.1815593 0.1955971 
100 0.1544014 0.1645331 
          150 0.1359382 0.1447487 
          200 0.1193891 0.1273547 
          250 0.1043342 0.1116742 
          300 0.09071633 0.09756991 
          350 0.07861059 0.08506636 
          400 0.06809837 0.07421481 
          450 0.05920095 0.06502153 
          500 0.05185695 0.05741881 
          550 0.04593181 0.05126896 
          600 0.04124379 0.04638704 
          650 0.03759254 0.0425684 
          700 0.034782 0.03961181 
          750 0.032635 0.03733479 
          800 0.03100038 0.0355813 
          850 0.02975462 0.03422377 
          900 0.02880007 0.03316147 
          950 0.02806161 0.03231709 
          1000 0.02748264 0.03163272 
          1050 0.02702125 0.03106581 
          1100 0.02664687 0.03058564 
          1150 0.02633737 0.03017033 
          1200 0.02607689 0.02980444 
          1250 0.025854 0.02947713 
          1300 0.02566047 0.02918074 
          1350 0.02549033 0.02890983 
          1400 0.02533914 0.02866039 
          1450 0.02520355 0.02842944 
          1500 0.02508101 0.02821464 
          1550 0.0249695 0.02801411 
          1600 0.0248674 0.02782629 
          1650 0.02477342 0.02764986 
          1700 0.02468646 0.02748365 
          1750 0.02460564 0.02732667 
          1800 0.0245302 0.02717803 
          1850 0.0244595 0.02703695 
          1900 0.02439301 0.02690273 
          1950 0.02433024 0.02677474 
          2000 0.02427081 0.02665246 ")
  
  data.v <- read.table(text=data1, header=T)
  par(mar=c(5.1, 4.1, 4.1, 4.1))
  plot(x=data.v$i, y=10*(data.v$loss), type="o", col="blue", pch=16, 
       main="IRIS loss and mse by 2-layers DNN",
       ylim=c(0, 2.0),
       xlab="",
       ylab="",
       axe =F)
  lines(x=data.v$i, y=10*data.v$mse, type="o", col="red", pch=1)
  box()
  axis(1, at=seq(0,2000,by=200))
  axis(4, at=seq(0.2,2.0,by=0.1))
  axis(2, at=seq(0.2,2.0,by=0.1))
  mtext("training step", 1, line=3)
  mtext("loss of training set (*10)", 2, line=2.5)
  mtext("mse of testing set (*10)", 4, line=2)
  
  legend("bottomleft", 
         legend = c("loss", "mse"),
         pch = c(16,1),
         col = c("blue","red"),
         lwd=c(1,1)
  )
}