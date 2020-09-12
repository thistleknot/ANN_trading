#primary <- 'rstudio'
#slave <- 'rparallel4'
#IPs <- list(list(host=primary, user='rstudio', ncore=4), list(host=slave, user='rstudio', ncore=4))
#spec <- lapply(IPs, function(IP) rep(list(list(host=IP$host, user=IP$user)), IP$ncore))
#spec <- unlist(spec, recursive=FALSE)
#stopCluster(cl)
#cl <- makeCluster(master=primary, spec=spec)
#clusterEvalQ(cl, {
## set up each worker.  Could also use clusterExport()
library(readxl)
library(readr)
library(quantmod)
library(xts)
library(zoo)
library(forecast)
library(TTR)
library(caret)
library(nnet)
library(parallel)
library(pbmcapply)
library(lubridate)
library(PerformanceAnalytics)
library(data.table)
library(imputeTS)
library(neuralnet)
library(boot)
library(plyr)
library(FCNN4R)
library(RSNNS)
library(IKTrading)
library(NeuralSens)
library(NeuralNetTools)
ncores <- detectCores(all.tests = FALSE, logical = TRUE)
#system("mkdir -p /home/rstudio/dev-DailyStockReport")
#system("scp rstudio:/home/rstudio/dev-DailyStockReport/customRules.R /home/rstudio/dev-DailyStockReport")
#})

data <- read.csv(file="states.csv",header = TRUE)
data2 <- data[,-1]

data2 <- data[,c(3:10,2)]

nr <- nrow(data2)

#start at 200
#-23 due to lag of donchian channel
trainSetIndex <- (sample(1:(nr),(nr)*.8))
testSetIndex <- c(1:nr)[(1:nr) %in% c(trainSetIndex)==FALSE]

# split the dataset 90-10% ratio
#sorted list but missing some elements

trainingdata <- data2[trainSetIndex,]
#View(trainingdata)
colnames(trainingdata)

#double subset

trainIndex <- sort(sample(c(1:length(trainSetIndex)), length(trainSetIndex)*.9))

set.train <- trainingdata[trainIndex, ]
set.test <- trainingdata[-trainIndex, ]

#normalization
trainParam <- caret::preProcess(as.matrix(set.train))

#method 1
#https://medium.com/@salsabilabasalamah/cross-validation-of-an-artificial-neural-network-f72a879ea6d5
frmla <- as.formula(paste(colnames(set.train)[ncol(set.train)], paste(colnames(set.train)[1:(ncol(set.train)-1)], sep = "", 
                                                                      collapse = " + "), sep = " ~ "))

#mlp
if(FALSE)
{
  model <- mlp( predict(trainParam, set.train)[,1:(ncol(set.train)-1)], predict(trainParam, set.train)[,(ncol(set.train))], size=5, learnFuncParams=c(0.1), maxit = 50, inputsTest=predict(trainParam, set.test)[,1:(ncol(set.test)-1)], targetsTest=predict(trainParam, set.test)[,(ncol(set.test))]) 
  summary(model)
  model
  weightMatrix(model)
  extractNetInfo(model)
  
  par(mfrow=c(2,2))
  plotIterativeError(model)
  
  predictions <- predict(model,predict(trainParam, set.test)[,1:(ncol(set.test)-1)])
  
  plotRegressionError(predictions[,1], predict(trainParam, set.test)[,(ncol(set.test))])
  
  confusionMatrix(predict(trainParam, set.train)[,(ncol(set.train))],fitted.values(model))
  confusionMatrix(predict(trainParam, set.test)[,(ncol(set.test))],predictions)
  
  plotROC(fitted.values(model)[,1], predict(trainParam, set.train)[,(ncol(set.train))])
  plotROC(predictions[,1], predict(trainParam, set.test)[,(ncol(set.test))])
  
  #confusion matrix with 402040-method
  confusionMatrix(predict(trainParam, set.train)[,(ncol(set.train))], encodeClassLabels(fitted.values(model),
                                                                                        method="402040", l=0.4, h=0.6))
}

#holdoutSet

#method 2
#https://rpubs.com/sergiomora123/Bitcoin_nnet
#would like to collapse this to set.train but for some reason throws a warning message if I do that. (of course set.train is later)

# derive the best neural network model using rmse criteria 
#best.network<-matrix(c(5,0.5))
best.network<-matrix(c(5))
#https://stackoverflow.com/questions/17105979/i-get-error-error-in-nnet-defaultx-y-w-too-many-77031-weights-whi/171071200

#cross validated
#parallelize
#i is used for size
#j is for decay

#set.seed(5)
numResamples <- 5

#clusterExport(cl, ls(all.names=TRUE), envir = .GlobalEnv)
#View(trainingdata)
#rmses <- pbmclapply (5:(ncol(set.train)+1), function(i)

#if I use mclapply, some return nothing... mlp not thread safe?
rmses <- lapply (5:15, function(i) 
  #i=5
{
  print(i)
  
  #set.rmse <- matrix(NA,numResamples)
  
  #trainIndex <- sample(1:nrow(trainingdata), nrow(trainingdata)*.5)
  
  #View(trainingdata)
  #folds <- createFolds(1:nrow(trainingdata),k=numResamples)
  set.seed(i)
  folds=sample(rep(1:numResamples, length=nrow(trainingdata)))
  
  set.rmse <- lapply (1:numResamples, function(x)
  {#x=3
    #print(x)
    set.train <- trainingdata[folds!=x,]
    set.test <- trainingdata[folds==x,]
    
    #normalization
    trainParam <- caret::preProcess(as.matrix(set.train))
    #summary(trainParam$mean)
    
    #j=1
    
    #/3 to avoid it growing too large
    #trainNN <- neuralnet(frmla, (predict(trainParam, set.train)), hidden = i , linear.output = T, stepmax = 1e5, algorithm='rprop-')
    
    trainNN <- mlp((predict(trainParam, set.train))[,1:(ncol(set.train)-1)], (predict(trainParam, set.train))[,(ncol(set.train))], size=i, learnFunc =  "SCG", linOut = TRUE, maxit = 250, inputsTest=predict(trainParam, set.test)[,1:(ncol(set.test)-1)], targetsTest=predict(trainParam, set.test)[,(ncol(set.test))]) 
    #trainNN$fittedTestValues
    #set.fit <- nnet(frmla, data = set.train, 
    #maxit=1000, MaxNWts=84581, size=i, decay=0.01*j, linout = 1)
    #maxit=1000, size=i, MaxNWts=(ncol(data2))^(i+1), decay=0.01*j, linout = 1)
    
    #this is a pdf
    #used with neuralnet
    #predict_testNN = compute(trainNN, (predict(trainParam, set.test)[,1:(ncol(set.test)-1)]))
    #used with mlp
    #same as trainNN$fittedTestValues
    predict_testNN <- predict(trainNN,(predict(trainParam, set.test)[,1:(ncol(set.test)-1)]))
    
    #convert back
    #inverse of log
    #used with neuralnet
    #pred <- (predict_testNN$net.result)*sd(set.train[,ncol(set.train)])+mean(set.train[,ncol(set.train)])
    pred <- (predict_testNN)*sd(set.train[,ncol(set.train)])+mean(set.train[,ncol(set.train)])
    #print(pred)
    pred[pred>0,] <- exp(1)^log(pred[pred>0,])
    pred[pred<0] <- -exp(1)^log(abs(pred[pred<0,]))
    #print(pred)
    
    denormalizedTrainPredictions <- pred
    #plot(denormalizedTrainPredictions,predict(trainParam,set.test)[,ncol(set.test)])
    
    #this is not normalized to a cdf
    return(sum((predict(trainParam,set.test)[,ncol(set.test)]*sd(set.train[,ncol(set.train)])+mean(set.train[,ncol(set.train)]) - denormalizedTrainPredictions)^2) / nrow(set.test)) ^ 0.5
  }#,mc.cores=(ncores)
  )
  meanrmse <- mean(unlist(set.rmse))
  return(list(i,meanrmse))
  
}#,mc.cores=(ncores)
)


best.rmse<-max(unlist(lapply(rmses, `[[`, 2)))
for(i in 1:length(rmses))
{#i=1
  print(i)
  if (rmses[[i]][[2]]<best.rmse) {
    best.network[1]<-rmses[[i]][[1]]
    #best.network[2,1]<-j
    best.rmse<-rmses[[i]][[2]]
  }
}

plot(unlist(lapply(rmses, `[[`, 1)),unlist(lapply(rmses, `[[`, 2)),type="l")

best.network
# create the Input and Target matrix for test
#best.network <-7
Testdata<-data2[testSetIndex,]

# fit the best model on test data
#set.fit <- nnet(frmla, data = trainingdata, maxit=1000, MaxNWts=84581, size=best.network[1,1], decay=0.1*best.network[2,1], linout = 1) 

traindataParam <- caret::preProcess(as.matrix(trainingdata))

#clusterExport(cl, ls(all.names=TRUE), envir = .GlobalEnv)
#candidate for parallelization
# repeat and average the model 20 times  

#set.fit <- mlp((predict(traindataParam, trainingdata))[,1:(ncol(set.train)-1)], (predict(traindataParam, trainingdata))[,(ncol(set.train))], size=best.network, learnFunc =  "Quickprop", learnFuncParams=c(0.1), maxit = 200)

#set.fit <- RSNNS::mlp(x = (predict(traindataParam, trainingdata))[,1:(ncol(set.train)-1)],y = (predict(traindataParam, trainingdata))[,(ncol(set.train))],size = best.network,linOut = TRUE,learnFunc =  "Quickprop", learnFuncParams=c(0.1),maxit=200)


#doesn't like mlp
#sensHess <- HessianMLP(set.fit, trData = (predict(traindataParam, trainingdata)), output_name = "Return")
set.fit <- nnet(frmla,data = (predict(traindataParam, trainingdata)),linear.output = T,size = best.network,maxit = 200)
sens <- SensAnalysisMLP(set.fit, trData = (predict(traindataParam, trainingdata)))

data_reduced <- data2

rmses=matrix(NA, ncol(data2)-2,2)
#rmses=matrix(NA, ncol(data2)-2)

numResamples=10

#reduce inputs
for(i in 1:(ncol(data2)-2))
{#i=1
  
  trainingdata <- data_reduced[trainSetIndex,]
  print(ncol(data_reduced))
  print(colnames(data_reduced))
  
  frmla <- as.formula(paste(colnames(trainingdata)[ncol(trainingdata)], paste(colnames(trainingdata)[1:(ncol(trainingdata)-1)], sep = "", 
                                                                              collapse = " + "), sep = " ~ "))
  
  traindataParam <- caret::preProcess(as.matrix(trainingdata))
  set.fit <- nnet(frmla,data = (predict(traindataParam, trainingdata)),linear.output = T,size = best.network,maxit = 200)
  
  #set.fit <- nnet(frmla, data = (predict(traindataParam, trainingdata)), maxit=200, decay=set.fitp$decay, size=best.network, linout = 1) 
  sens <- SensAnalysisMLP(set.fit, trData = (predict(traindataParam, trainingdata)),plot=FALSE)
  
  set.seed(i)
  folds=sample(rep(1:numResamples, length=nrow(trainingdata)))
  
  set.rmse <- lapply (1:numResamples, function(x)
  {#x=3
    #print(x)
    set.train <- trainingdata[folds!=x,]
    set.test <- trainingdata[folds==x,]
    
    trainParam <- caret::preProcess(as.matrix(set.train))
    
    trainNN <- mlp((predict(trainParam, set.train))[,1:(ncol(set.train)-1)], (predict(trainParam, set.train))[,(ncol(set.train))], size=best.network, learnFunc =  "SCG", linOut = TRUE, maxit = 250, inputsTest=predict(trainParam, set.test)[,1:(ncol(set.test)-1)], targetsTest=predict(trainParam, set.test)[,(ncol(set.test))]) 
    
    predict_testNN <- predict(trainNN,(predict(trainParam, set.test)[,1:(ncol(set.test)-1)]))
    
    pred <- (predict_testNN)*sd(set.train[,ncol(set.train)])+mean(set.train[,ncol(set.train)])
    
    pred[pred>0,] <- exp(1)^log(pred[pred>0,])
    pred[pred<0] <- -exp(1)^log(abs(pred[pred<0,]))
    
    denormalizedTrainPredictions <- pred
    return(sum((predict(trainParam,set.test)[,ncol(set.test)]*sd(set.train[,ncol(set.train)])+mean(set.train[,ncol(set.train)]) - denormalizedTrainPredictions)^2) / nrow(set.test)) ^ 0.5
  }
  )
  meanrmse <- mean(unlist(set.rmse))
  print(meanrmse)
  
  #exclude <- rownames(sens$sens$.outcome[sens$sens$.outcome$meanSensSQ==max(sens$sens$.outcome$meanSensSQ),])
  exclude <- rownames(data.frame(garson(set.fit,bar_plot=F)))[garson(set.fit,bar_plot=F)==(min(garson(set.fit,bar_plot=F)))]
  
  rmses[i,] <- c(meanrmse,paste(colnames(data_reduced), collapse=" + "))
  #rmses[i] <- meanrmse
  print(exclude)
  
  data_reduced <- data_reduced[ , -which(colnames(data_reduced) %in% c(exclude))]
  SensitivityPlots(sens)
  garson(set.fit)
  
  
  
}
plot(rmses[,1],type="l")
newForm <- rmses[which(rmses[,1]==min(rmses[,1])),2]
print(rmses[which(rmses[,1]==min(rmses[,1])),2])

newFrmla <- (t(read.csv(text=gsub(" \\+ ", ",", newForm),header=F)))

newData <- data2[,c(newFrmla)]

trainingdata <- newData[trainSetIndex,]
Testdata<-newData[testSetIndex,]

traindataParam <- caret::preProcess(as.matrix(trainingdata))

set.predict <- lapply (1:5, function(x) {
  #x=1
  set.fit <- mlp((predict(traindataParam, trainingdata))[,1:(ncol(trainingdata)-1)], (predict(traindataParam, trainingdata))[,(ncol(trainingdata))], size=best.network, linOut = TRUE, learnFunc =  "SCG", maxit = 250) 
  
  predict(set.fit, predict(traindataParam, data.frame(Testdata))[,1:(ncol(Testdata)-1)],2,)
  
}#,mc.cores=(ncores)
)
#stopCluster(cl)
set.predict1 <- data.frame(rowMeans(do.call(cbind, set.predict)))

set.predict1 <- set.predict1*sd(trainingdata[,ncol(trainingdata)])+mean(trainingdata[,ncol(trainingdata)])

plot(unlist(Testdata[,ncol(Testdata)]),unlist(set.predict1))
abline(lm(unlist(Testdata[,ncol(Testdata)])~unlist(set.predict1)))

#apply to whole dataSet
traindataParam <- caret::preProcess(as.matrix(newData))

set.fit <- RSNNS::mlp((predict(traindataParam, newData))[,1:(ncol(newData)-1)], (predict(traindataParam, newData))[,(ncol(newData))], size=best.network, linOut = TRUE, learnFunc =  "SCG", maxit = 250)

garson(set.fit)
newFrmla <- as.formula(paste(colnames(newData)[ncol(newData)], paste(colnames(newData)[1:(ncol(newData)-1)], sep = "", 
                                                         collapse = " + "), sep = " ~ "))

set.fit <- nnet(newFrmla,data = predict(traindataParam, newData),linear.output = T,size = best.network,maxit = 300)
outputC <- paste0(colnames(newData[,ncol(newData),drop=F]))
sens <- SensAnalysisMLP(set.fit, trData = predict(traindataParam, newData), output_name = paste0(colnames(newData[,ncol(newData),drop=F])))

prediction <- set.fit$fitted.values*traindataParam$std[ncol(newData)]+traindataParam$mean[ncol(newData)]
actual <- newData[,ncol(newData)]
plot(prediction,actual)
abline(lm(actual~prediction))