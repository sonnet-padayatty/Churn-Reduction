rm(list=ls(all=T))
setwd("E:/project")

#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')
lapply(x, require, character.only = TRUE)
rm(x)

## Read the data
churn = read.csv("train_data.csv", header = T, na.strings = c(" ", "", "NA"))
churn_test=read.csv('test_data.csv',header = T, na.strings = c(" ", "", "NA"))
df=rbind(churn,churn_test)
View(df)
churn=df
str(churn)
churn$area.code=as.factor(churn$area.code)

##################################Missing Values Analysis###############################################
missing_val = data.frame(apply(churn,2,function(x){sum(is.na(x))}))
missing_val


##Data Manupulation; convert string categories into factor numeric
for(i in 1:ncol(churn)){
  
  if(class(churn[,i]) == 'factor'){
    
    churn[,i] = factor(churn[,i], labels=(1:length(levels(factor(churn[,i])))))
    
  }
}

############################################Outlier Analysis#############################################
# ## BoxPlots - Distribution and Outlier Check

numeric_index = sapply(churn,is.numeric) #selecting only numeric

numeric_data = churn[,numeric_index]

cnames = colnames(numeric_data)
cnames



for (i in 1:length(cnames))
   {
     assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "Churn"), data = subset(churn))+ 
              stat_boxplot(geom = "errorbar", width = 0.5) +
              geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                           outlier.size=1, notch=FALSE) +
              theme(legend.position="bottom")+
              labs(y=cnames[i],x="Churn")+
              ggtitle(paste("Box plot of churn for",cnames[i])))
   }   

### Plotting plots together
 gridExtra::grid.arrange(gn1,gn5,gn2,ncol=3)
 gridExtra::grid.arrange(gn6,gn7,gn3,ncol=3)
 gridExtra::grid.arrange(gn8,gn9,gn4,ncol=3)
 gridExtra::grid.arrange(gn10,gn11,gn12,ncol=3)
 gridExtra::grid.arrange(gn13,gn14,gn15,ncol=3)
 
 
 
 sum(is.na(churn))
# #Replace all outliers with NA and impute

 for(i in cnames){
   val = churn[,i][churn[,i] %in% boxplot.stats(churn[,i])$out]
   churn[,i][churn[,i] %in% val] = NA
 }
 
 churn = knnImputation(churn, k = 5)

write.csv(churn,'merged_without_outliers.csv', row.names = F)

##################################Feature Selection################################################
## Correlation Plot 
corrgram(churn[,numeric_index], order = F,
         upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")


## Chi-squared Test of Independence
factor_index = sapply(churn,is.factor)
factor_data = churn[,factor_index]
View(factor_data)
for (i in 1:5)
{
  print(names(factor_data)[i])
  print(chisq.test(table(factor_data$Churn,factor_data[,i])))
}

## Dimension Reduction
churn = subset(churn, 
                     select = -c(area.code,total.day.charge,total.eve.charge, total.night.charge, total.intl.charge,phone.number))

##################################Feature Scaling################################################
#Normality check

qqnorm(churn$number.customer.service.calls)
hist(churn$number.customer.service.calls)
#Normalisation
cnames1= c("number.vmail.messages","number.customer.service.calls")

for(i in cnames1){
  churn[,i] = (churn[,i] - min(churn[,i]))/
    (max(churn[,i] - min(churn[,i])))
}


 #Standardisation

cnames2=c("account.length","total.day.minutes","total.day.calls",
          "total.intl.minutes","total.intl.calls","total.eve.minutes",
          "total.eve.calls","total.night.minutes","total.night.calls")
 
for(i in cnames2){
  churn[,i] = (churn[,i] - mean(churn[,i]))/
                                  sd(churn[,i])
}
View(churn)


############################################################################
#######Import test data #######################

## Read the data
churn_test=churn[3334:5000,]
churn=churn[1:3333,]
View(churn_test)

###################################Model Development#######################################
#Clean the environment
rmExcept(c("churn","churn_test"))

#Decision tree for classification
#Develop Model on training data
C50_model = C5.0(Churn ~.,churn,trials=100,rules = TRUE)

#Summary of DT model
summary(C50_model)

#write rules into disk
write(capture.output(summary(C50_model)), "c50mergedrules.txt")


#Lets predict for test cases
C50_Predictions = predict(C50_model, churn_test[,-15], type = "class")

#write predicted output into disk
write(capture.output(C50_Predictions), "Predicted_output.txt")


##Evaluate the performance of classification model
ConfMatrix_C50 = table(churn_test$Churn, C50_Predictions)
confusionMatrix(ConfMatrix_C50)


#Accuracy: 94.48%
#FNR: 40.2%
##################################################################################
###Random Forest
RF_model = randomForest(Churn ~ ., churn, importance = TRUE, ntree = 500)

#Presdict test data using random forest model
RF_Predictions = predict(RF_model, churn_test[,-15])

##Evaluate the performance of classification model
ConfMatrix_RF = table(churn_test$Churn, RF_Predictions)
confusionMatrix(ConfMatrix_RF)
#Accuracy=92.3%
#FNR=51%

##############################################################################
#Logistic Regression
logit_model = glm(Churn ~ ., data = churn, family = "binomial")

#summary of the model
summary(logit_model)

#predict using logistic regression
logit_Predictions = predict(logit_model, newdata = churn_test, type = "response")

#convert prob
logit_Predictions = ifelse(logit_Predictions > 0.5, 1, 0)


##Evaluate the performance of classification model
ConfMatrix_RF = table(churn_test$Churn, logit_Predictions)
ConfMatrix_RF
#accuracy=87.4%
#False Negative rate=80.8%

######################################################################

##KNN Implementation
library(class)

#Predict test data
KNN_Predictions = knn(churn[, 1:14], churn_test[, 1:14], churn$Churn, k = 7)

#Confusion matrix
Conf_matrix = table(KNN_Predictions, churn_test$Churn)
Conf_matrix
#Accuracy
sum(diag(Conf_matrix))/nrow(churn_test)


#Accuracy = 86.6%
#FNR = 46.6%
################################################################################

#naive Bayes
library(e1071)

#Develop model
NB_model = naiveBayes(Churn ~ ., data = churn)

#predict on test cases #raw
NB_Predictions = predict(NB_model, churn_test[,1:14], type = 'class')

#Look at confusion matrix
Conf_matrix = table(observed = churn_test[,15], predicted = NB_Predictions)
confusionMatrix(Conf_matrix)

#Accuracy: 88%
#FNR: 79%

