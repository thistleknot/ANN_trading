#https://rpubs.com/sergiomora123/Bitcoin_nnet

run_librarys <- function() {
  library(readxl)
  library(readr)
  library(quantmod)
  library(xts)
  library(zoo)
  library(forecast)
  #library(quantstrat)
  #library(Quandl)
  #library(PerformanceAnalytics)
  library(TTR)
  library(caret)
  library(nnet)
  library(lubridate)
  library(data.table)
  library(imputeTS)
  print("end........")
}
run_librarys()

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
  ans = rollapply(x,26,FUN = autoregressor1,by.column=FALSE)
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

symbolstring1 <- c('^SP500TR','GOOG','SSO','GOLD')

getSymbols(symbolstring1,from=startDate,to=endDate,adjust=TRUE,src='yahoo')

symbolstring1 <- c('SP500TR','GOOG','SSO','GOLD')
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

chosen <- "GOLD"
data <- get(chosen)

data2<-data

price<-(data2$Close)
HLC<-matrix(c(data2$High, data2$Low, data2$Close),nrow=length(data2$High))

# calculate log returns
set.lr<-diff(log(price))

# generate technical indicators 
rsi<-RSI(price)
MACD <- MACD(price)
macd<-MACD[,1]
will<-williamsAD(HLC)
cci<-CCI(HLC)
STOCH<-stoch(HLC)
stochK<-STOCH[,1]
stochD<-STOCH[,1]
#ar <- autoregressor(price)
ar <- autoregressor(WMA(data2[,"Adjusted"], wts = data2[,"Volume"]))

nr <- nrow(HLC)
size <- 939-700
#size <- 939-500
ud2 <- 970-941
upper <- nrow(HLC)-ud2-1
#upper <- nr-size
lower <- upper-size
d <- 940-917
ud <- 969-940

#Input<-(matrix(c(rownames(data.frame(rsi[lower:upper,,drop=FALSE])),rsi[lower:upper], cci[lower:upper], macd[lower:upper], will[lower:upper], stochK[lower:upper], stochD[lower:upper]),nrow=240)[,-1])

Input<-
  (
    matrix(
      cbind(rsi[lower:upper], cci[lower:upper], macd[lower:upper], will[lower:upper], stochK[lower:upper], stochD[lower:upper], ar[lower:upper])
      ,nrow=size+1)
  )

Target<-as.numeric(matrix(c(set.lr[(lower+1):(upper+1)]), nrow=size+1))

trainingdata <- cbind(Input,Target)
#View(trainingdata)

colnames(trainingdata) <- c("RSI","CCI","MACD","WILL","STOCHK","STOCHD", "AR", "Return")

# split the dataset 90-10% ratio
trainIndex <- createDataPartition(set.lr[(lower+1):(upper+1)], p=.9, list=F)
set.train <- trainingdata[trainIndex, ]
set.test <- trainingdata[-trainIndex, ]

InputTest<-matrix(cbind(rsi[upper:(upper+ud)], cci[upper:(upper+ud)], macd[upper:(upper+ud)], will[upper:(upper+ud)], stochK[upper:(upper+ud)], stochD[upper:(upper+ud)], ar[upper:(upper+ud)]),nrow=ud+1)
TargetTest<-matrix(c(set.lr[(upper+1):(upper+1+ud2)]), nrow=ud+1)

Testdata <- cbind(InputTest,TargetTest)
colnames(Testdata) <- c("RSI","CCI","MACD","WILL","STOCHK","STOCHD", "AR", "Return")

set.predict1 <- Testdata[,c("AR")]

#rules
#sell
#set.predict1[set.predict1>.25]=-1
#set.predict1[set.predict1<-.25]=1

set.predict1[set.predict1>0]=-1
set.predict1[set.predict1<0]=1

money<-matrix(0,31)
#money2 is benchmark, actual price
money2<-matrix(0,31)
money[1,1]<-100
money2[1,1]<-100
for (i in 2:31) {
  #print(i)
  #i=3
  #if (abs(set.predict1[i-1])==1) {
  if ((set.predict1[i-1])<0) {
    #direction1<-set.predict[i-1]
    direction1<--1
  } else {
    #direction1<-0}
    direction1<-1}
  if (TargetTest[i-1]<0) {
    direction2<--1  
  } else {
    direction2<-1 }
  #if ((direction1-direction2)==0) {
  if ((direction1-direction2)==0) {
    money[i,1]<-money[i-1,1]*(1+abs(TargetTest[i-1]))  
  } else {
    money[i,1]<-money[i-1,1]*(1-abs(TargetTest[i-1])) }
  money2[i,1]<-100*(as.numeric(price[upper+i-1])/as.numeric(price[upper]))
}

#plot benchmark and neural network profit on the test dataset
x<-1:31
matplot(cbind(money, money2), type = "l", xaxt = "n", ylab = "")
legend("topright", legend = c("Neural network","Benchmark"), pch = 19, col = c("black", "red"))
axis(1, at = c(1,10,20,ud+1), lab  = row.names(data.frame(rsi[upper:(upper+ud),,drop=FALSE]))[c(1,10,20,ud+1)])

box()
mtext(side = 1, "Test dataset", line = 2)
mtext(side = 2, "Investment value", line = 2)