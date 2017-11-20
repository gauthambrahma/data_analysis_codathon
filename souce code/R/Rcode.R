#installing the packages
install.packages("rpart.plot") 
#importing the packages
library("rpart")
library ("rpart.plot")
#reading data
data2=read.table(file="http://www.cc.gatech.edu/gvu/user_surveys/survey-1997-10/datasets/final_privacy.repl",header = TRUE,sep = "\t",quote = "\"")
#splitting the data into training and testing sets
samples<-sample(2,nrow(data2),replace = TRUE,prob = c(0.7,0.3))
trainingData<-data2[samples==1,]
trainingData=lapply(trainingData,function(x)as.numeric(as.factor(x)))
testingData<-data2[samples==2,]
temp<-data2[samples==2,]
testingData=lapply(testingData,function(x)as.numeric(as.factor(x)))
#chi square values for all comuns with respect to first column
pvaluesList<-list()
for(colnumber in 2:ncol(trainingDataM)){
     pvaluesList[[colnumber]]<-chisq.test(unlist(trainingData[1]),unlist(trainingData[colnumber]))$p.value
}
trainingData<-data.frame(trainingData[1],trainingData[23],trainingData[9],trainingData[4],trainingData[3],trainingData[8])
testingData<-data.frame(testingData[23],testingData[9],testingData[4],testingData[3],testingData[8])
#building a desicion tree
fit <- rpart(Advertising.Networks ~ ., method="class", data=trainingData, control=rpart.control(minsplit=30,cp=0.001))
tpredict<-predict(fit, newdata=testingData, type="class",parms = list(split = "information"))
confusionMatrix<-table(tpredict,temp$Advertising.Networks)
accuracy <- sum(diag(confusionMatrix))/sum(confusionMatrix)
#end of classification below is the code for regression. Some data variables may be used from above
library (lattice)
lrresults <- lm (Advertising.Networks ~ ., data = trainingData)
summary(lrresults)
confint(lrresults,level = .95)
conf_int_pt<-predict(lrresults, trainingData, level=.95, interval= "confidence")
