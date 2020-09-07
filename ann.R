run_librarys <- function() {
  library(readxl)
  library(readr)
  library(quantmod)
  library(xts)
  library(zoo)
  library(forecast)
  #library(quantstrat)
  #library(Quandl)
  library(TTR)
  library(caret)
  library(nnet)
  library(parallel)
  library(lubridate)
  library(PerformanceAnalytics)
  library(data.table)
  library(imputeTS)
  library(neuralnet)
  library(boot)
  library(plyr)
  library(FCNN4R)
  library(RSNNS)
  print("end........")
}
run_librarys()

ncores <- detectCores(all.tests = FALSE, logical = TRUE)

source("~/ANN/TradingDates.R")

autoregressor1  = function(x){
  if(NROW(x)<12){ result = NA} else{
    #y = ROC(Ad(x))
    y = ROC(x)
    y = na.omit(y)
    step1 = ar.yw(y)
    step2 = predict(step1,newdata=y,n.ahead=1)
    step3 = step2$pred[1]+1
    #step4 = (step3*last(((Ad(x)*Vo(x))))) - last(((Ad(x)*Vo(x))))
    step4 = (step3*last(x)) - last(x)
    
    result = step4
  }
  return(result)
}

autoregressor = function(x){
  ans = rollapply(x,28,FUN = autoregressor1,by.column=FALSE)
  return (ans)}


extract_year <- function (x_date)
{
  #https://stackoverflow.com/questions/36568070/extract-year-from-date
  substring(x_date,1,4) #This takes string and only keeps the characters beginning in position 7 to position 10
}

endDate <- as.Date(Sys.Date())
startDate <- endDate %m-% years(3)

years <- (as.integer(extract_year((startDate))):as.integer(extract_year(endDate)))

dates <- lapply(years, function (x){
  TradingDates(x)
})

trading_dates <- data.frame(as.Date(unlist(dates)))[,,drop=FALSE]
colnames(trading_dates) <- "Date"
rownames(trading_dates) <- trading_dates$Date

trading_dates <- subset(trading_dates, rownames(trading_dates) <=  endDate & rownames(trading_dates) >= startDate)

symbolstring1 <- c('^GSPC','GOOG','SSO','GOLD')

getSymbols(symbolstring1,from=startDate,to=endDate,adjust=TRUE,src='yahoo')

symbolstring1 <- c('GSPC','GOOG','SSO','GOLD')

#symbolstring1 <- c('SP500TR','GOLD')

for(it in 1:length(symbolstring1))
{#it=1
  
  temp <- get(symbolstring1[it])
  colnames(temp) <- c("Open","High","Low","Close","Volume","Adjusted")
  rm(list=symbolstring1[it])
  assign(symbolstring1[it],temp)
}

#adjusted
for(it in 1:length(symbolstring1))
{x=symbolstring1[it]
set <- data.frame(quantmod::adjustOHLC(as.xts(as.data.table(get(x)[,c("Open","High","Low","Close","Volume","Adjusted")])),use.Adjusted=TRUE,symbol.name=x))
d <- data.frame(as.Date(rownames(set)))[,,drop=FALSE]
colnames(d) <- "Date"
set <- cbind(d,set)

temp <- set
rm(list=symbolstring1[it])
assign(symbolstring1[it],temp)
rm(set)
rm(temp)
rm(d)
}

adjustedDF <- mclapply(symbolstring1, function(x)
{
  return(get(x))
})

names(adjustedDF) <- symbolstring1

adjustedDF <- rbindlist(as.list(adjustedDF),idcol="Symbol")

adjusted_pvt <- reshape2::dcast(adjustedDF, Date ~ Symbol,value.var='Adjusted',fun.aggregate = mean, fill=NULL)
adjusted_pvt <- merge.data.frame(x=trading_dates[,'Date',drop=F],y=data.frame(adjusted_pvt),by='Date',all.x=T)
adjusted_pvt <- na_interpolation(adjusted_pvt,options=LINEAR)

rownames(adjusted_pvt) <- adjusted_pvt$Date
#uniques <- length(unique(combined[,y]))
#if(uniques>=5){uniques=0}

adjusted_pvt_returns <- CalculateReturns(adjusted_pvt[,2:ncol(adjusted_pvt)], method="discrete")
rownames(adjusted_pvt_returns) <- adjusted_pvt$Date
adjusted_pvt_returns <- tail(adjusted_pvt_returns,-1)

#remove date, set as.xts
for(it in 1:length(symbolstring1))
{x=symbolstring1[it]

inter <- merge.data.frame(x=trading_dates[,'Date',drop=F],y=data.frame(get(x)),by='Date',all.x=T)
inter <- na_interpolation(inter,options=LINEAR)

rownames(inter) <- inter$Date

temp <- as.xts(get(x)[c("Open","High","Low","Close","Volume","Adjusted")])

rm(list=symbolstring1[it])
assign(symbolstring1[it],temp)
rm(temp)
}

#this is where I'll parallelize

chosen <- "GOOG"
data <- get(chosen)

data2<-data

price<-(data2$Close)
HLC<-matrix(c(data2$High, data2$Low, data2$Close),nrow=length(data2$High))

# calculate log returns
set.lr<-diff(log(price))
#set.lr<-diff((price))

# generate technical indicators 
rsi<-RSI(price)
MACD <- MACD(price)
macd<-MACD[,1]
will<-williamsAD(HLC)
cci<-CCI(HLC)
STOCH<-stoch(HLC)
stochK<-STOCH[,1]
stochD<-STOCH[,2]
ar <- autoregressor(WMA(data2[,"Adjusted"], wts = data2[,"Volume"]))
sar <- TTR::SAR(HLC)
colnames(sar) <- c("SAR")
cmf <- CMF(data2, volume=data2$Volume,n=20)
colnames(cmf) <- c("CMF")
bbands <- BBands(HLC, volume=data$Volume, n=20, maType=VWAP, sd=2)
nextDay <- matrix(lag(set.lr,1))
colnames(nextDay) <- c("Return")

TTRs <- cbind(rsi, macd, will, cci, stochK, stochD, ar, sar, cmf, bbands, nextDay)

nr <- nrow(price)

#start at 28
trainSetIndex <- sort(sample(28:(nr-1),nrow(price[28:(nr-1)])*.8))
testSetIndex <- c(28:(nr-1))[(c(28:(nr-1))) %in% c(trainSetIndex)==FALSE]
  
# split the dataset 90-10% ratio
#sorted list but missing some elements

trainingdata <- TTRs[trainSetIndex,]

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
  model = model <- mlp( predict(trainParam, set.train)[,1:(ncol(set.train)-1)], predict(trainParam, set.train)[,(ncol(set.train))], size=5, learnFuncParams=c(0.1), maxit = 50, inputsTest=predict(trainParam, set.test)[,1:(ncol(set.test)-1)], targetsTest=predict(trainParam, set.test)[,(ncol(set.test))]) 
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

Testdata <- TTRs[28:nrow(TTRs),][-trainSetIndex,]

#method 2
#https://rpubs.com/sergiomora123/Bitcoin_nnet
#would like to collapse this to set.train but for some reason throws a warning message if I do that. (of course set.train is later)

# derive the best neural network model using rmse criteria 
#best.network<-matrix(c(5,0.5))
best.network<-matrix(c(5))
best.rmse<-1
#https://stackoverflow.com/questions/17105979/i-get-error-error-in-nnet-defaultx-y-w-too-many-77031-weights-whi/17107128

#cross validated
#parallelize
#i is used for size
#j is for decay
set.seed(5)
for (i in 5:(ncol(set.train)+1)) 
  #i=15
{
  print(i)
  numResamples <- 5
  #set.rmse <- matrix(NA,numResamples)
  
  set.rmse <- mclapply (1:numResamples, function(x)
    {#j=4
    #print(j)
  
    trainIndex <- sample(1:nrow(trainingdata), nrow(trainingdata)*.9)
    
    set.train <- trainingdata[trainIndex, ]
    set.test <- trainingdata[-trainIndex, ]
    
    #normalization
    trainParam <- caret::preProcess(as.matrix(set.train))
    
    #j=1
    
    #/3 to avoid it growing too large
    trainNN <- neuralnet(frmla, predict(trainParam, set.train), hidden = i , linear.output = T, stepmax = 1e7, algorithm='rprop-')
  
    #set.fit <- nnet(frmla, data = set.train, 
                        #maxit=1000, MaxNWts=84581, size=i, decay=0.01*j, linout = 1)
                    #maxit=1000, size=i, MaxNWts=(ncol(data2))^(i+1), decay=0.01*j, linout = 1)
    
    predict_testNN = compute(trainNN, predict(trainParam, set.test)[,1:(ncol(set.test)-1)])
    
    #convert back
    #inverse of log
    
    pred <- predict_testNN$net.result*sd(set.train[,ncol(set.train)])+mean(set.train[,ncol(set.train)])
    pred[pred>0,] <- exp(1)^log(pred[pred>0,])
    pred[pred<0] <- -exp(1)^log(abs(pred[pred<0,]))
    
    denormalizedTrainPredictions <- pred
    
    return(sum((predict(trainParam,set.test)[,ncol(set.test)]*sd(set.train[,ncol(set.train)])+mean(set.train[,ncol(set.train)]) - denormalizedTrainPredictions)^2) / nrow(set.test)) ^ 0.5
  },mc.cores=ncores)
  meanrmse <- mean(unlist(set.rmse))
  if (meanrmse<best.rmse) {
    best.network[1]<-i
    #best.network[2,1]<-j
    best.rmse<-meanrmse  
    
  }
}
# create the Input and Target matrix for test
#best.network <-7
Testdata<-TTRs[testSetIndex,]

# fit the best model on test data
#set.fit <- nnet(frmla, data = trainingdata, maxit=1000, MaxNWts=84581, size=best.network[1,1], decay=0.1*best.network[2,1], linout = 1) 

traindataParam <- caret::preProcess(as.matrix(trainingdata))

#candidate for parallelization
# repeat and average the model 20 times  
set.predict <- mclapply (1:5, function(x) {
  #set.fit <- nnet(frmla, data = trainingdata, maxit=1000, size=best.network[1,1], decay=0.1*best.network[2,1], linout = 1) 
  set.fit <- neuralnet(frmla, predict(traindataParam, trainingdata), hidden = best.network , linear.output = T, stepmax = 1e6, algorithm='rprop-')
  
  #return(
    predict(set.fit, newdata = predict(traindataParam, data.frame(Testdata))[,1:(ncol(Testdata)-1)])
    #)
  
},mc.cores=ncores)

set.predict1 <- data.frame(rowMeans(do.call(cbind, set.predict)))

set.predict1 <- set.predict1*sd(trainingdata[,ncol(trainingdata)])+mean(trainingdata[,ncol(trainingdata)])

pred <- set.predict1

pred[pred>0] <- exp(1)^log(pred[pred>0,])
pred[pred<0] <- -exp(1)^log(abs(pred[pred<0,]))

set.predict1 <- pred

prep <- Testdata[,"Return"]

pred <- data.frame(prep)
pred[pred>0,] <- exp(1)^log(pred[pred>0,])
pred[pred<0] <- -exp(1)^log(abs(pred[pred<0,]))

invertLogTest <- pred

plot(unlist(invertLogTest),unlist(set.predict1))
abline(lm(unlist(invertLogTest)~unlist(set.predict1)))

# calculate the buy-and-hold benchmark strategy and neural network profit on the test dataset
money<-matrix(0,31)
#money2 is benchmark, actual price
money2<-matrix(0,31)
money[1,1]<-100
money2[1,1]<-100

for (i in 2:31) {
  #print(i)
  #i=4
  
  if (set.predict1[i-1,]<0) {
    direction1<--1  
  } else {
    direction1<-1}
  if (Testdata[i-1,"Return",]<0) {
    direction2<--1  
  } else {
    direction2<-1 }
  if ((direction1-direction2)==0) {
    
    #gain return on correct guesses
    money[i,1]<-money[i-1,1]*(1+abs(Testdata[i-1,"Return",]))  
  } else {
    
    #lose return on incorrect guesses, but... incorrect guesses is when it goes down.
    money[i,1]<-money[i-1,1]*(1-abs(Testdata[i-1,"Return",]*2)) }
  money2[i,1]<-100*(as.numeric(price[rownames(data.frame(Testdata[i-1,"Return",])),])/as.numeric(price[rownames(data.frame(Testdata[1,"Return",])),]))
}

#By always guessing 1, you match the market?
#cbind(money,money2)

#plot benchmark and neural network profit on the test dataset
x<-1:31
matplot(cbind(money, money2), type = "l", xaxt = "n", ylab = "")
legend("topright", legend = c("Neural network","Benchmark"), pch = 19, col = c("black", "red"))
axis(1, at = c(1,10,20,holdOutSize+1), lab  = row.names(data.frame(rsi[upper:(upper+holdOutSize),,drop=FALSE]))[c(1,10,20,holdOutSize+1)])

box()
mtext(side = 1, "Test dataset", line = 2)
mtext(side = 2, "Investment value", line = 2)