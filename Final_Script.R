#Reading CSV files
train=read.csv("train2016.csv")
test=read.csv("test2016.csv")

#Imputation of the missing Demographic Data
library(mice)
train$YOB[train$YOB>2003 | train$YOB<1935]<-NA
imputeVar=setdiff(names(train),"Party")
imputed=complete(mice(train[imputeVar]))
train$YOB=imputed$YOB
test$YOB[test$YOB>2003 | test$YOB<1935]<-NA
imputeTest=complete(mice(test[names(test)]))
test$YOB=imputeTest$YOB

#Making Age Groups from Year of Birth info for better predictions
train$age_grcut <- cut(train$YOB, breaks = c(-Inf, 1950,1965 ,1980,1995, Inf), labels = c("old", "46-60 yrs", "31-45 yrs", "18-30 yrs", "0-18 yrs"), right = FALSE)
test$age_grcut <- cut(test$YOB, breaks = c(-Inf, 1950,1965 ,1980,1995, Inf), labels = c("old", "46-60 yrs", "31-45 yrs", "18-30 yrs", "0-18 yrs"), right = FALSE)

#Spliting the train data for validation from our side
library(caTools)
spl=sample.split(train$Party,SplitRatio=0.55)
train2<-subset(train,spl==TRUE)
test2<-subset(train,spl==FALSE)

#Better prediction variables from Feature Engineering and other algorithms

formvars<-c("YOB","Gender","Income","HouseholdStatus","EducationLevel","age_grcut","Q124742","Q124122","Q122771","Q122120","Q121699","Q121700","Q120379","Q120650","Q120472","Q120194","Q119851","Q118232","Q118233","Q116881","Q116953","Q116197","Q115611","Q115899","Q115390","Q115195","Q114517","Q113583","Q113181","Q112478","Q111220","Q110740","Q108950","Q109244","Q108855","Q108342","Q106272","Q106389","Q106042","Q104996","Q102089","Q101163","Q100689","Q99480","Q98869","Q98578","Q98059","Q98197")

form<-Party~YOB+Gender+Income+HouseholdStatus+EducationLevel+age_grcut+Q124742+Q124122+Q122771+Q122120+Q121699+Q121700+Q120379+Q120650+Q120472+Q120194+Q119851+Q118232+Q118233+Q116881+Q116953+Q116197+Q115611+Q115899+Q115390+Q115195+Q114517+Q113583+Q113181+Q112478+Q111220+Q110740+Q108950+Q109244+Q108855+Q108342+Q106272+Q106389+Q106042+Q104996+Q102089+Q101163+Q100689+Q99480+Q98869+Q98578+Q98059+Q98197

#Logistic Regression
modelLog2=glm(form,data=train,family=binomial)
PredLog=predict(modelLog2,newdata=test,type="response")
table(test$Party,PredLog>=0.5)
threshold = 0.5
PredTestLabels = as.factor(ifelse(PredLog<threshold, "Democrat", "Republican"))
submitL=data.frame(USER_ID=test$USER_ID,Predictions=PredTestLabels)
write.csv(submitL, file =  "GLM.csv", row.names = FALSE)

#CART-model
library(rpart)
library(rpart.plot)

CART1=rpart(form,data=train,method="class")
Pred1=predict(CART1,newdata=test,type="class")
table(test2$Party,Pred1)
Prediction=predict(CART1,newdata=test,type="class")
submitA=data.frame(USER_ID=test$USER_ID,Predictions=Prediction)
write.csv(submitA, file =  "Cart.csv", row.names = FALSE)

#Random Forest Model
library(randomForest)
RF2=randomForest(form,data=train,cp=0.068)
PredRFTrain=predict(RF2,type="prob")[,2]
PredRF=predict(RF2,newdata=test,type="prob")[,2]
table(test2$Party,PredRF)
PredRFlabel = as.factor(ifelse(PredRF<threshold, "Democrat", "Republican"))
submitRF=data.frame(USER_ID=test$USER_ID,Predictions=PredRFlabel)
write.csv(submitRF, file =  "RandomForest.csv", row.names = FALSE)

#CForest Model
library(party)
CF2=cforest(form,data=train)
CF=cforest(Party~YOB+Gender+Income+HouseholdStatus+EducationLevel+Q118892+Q119851+Q121011+Q120650+Q120978+Q121700+Q118117+Q120012+Q124742+Q121699+Q121700+Q120194+Q118232+Q116197+Q115611+Q114517+Q113181+Q112478+Q111220+Q108950+Q109244+Q108342+Q102687+Q101596+Q100689+Q99716+Q98869+Q98578+Q98197,data=train)
PredCF=predict(CF2,newdata=test)
submitCF=data.frame(USER_ID=test$USER_ID,Predictions=PredCF)
write.csv(submitCF, file =  "CForest.csv", row.names = FALSE)


#Kmeans Clustering for segmenting data into different clusters
limTrain<-as.data.frame(lapply(train[names(train)%in%formvars],as.numeric))
limTest<-as.data.frame(lapply(test[names(test)%in%formvars],as.numeric))
limTrain$Party<-NULL

#Normalization
preproc=preProcess(limTrain)
normtrain=predict(preproc,limTrain)
normtest=predict(preproc,limTest)
kmc=kmeans(normtrain,centers = 5,iter.max=1000)

table(kmc$cluster)
library(flexclust)
km.kcca = as.kcca(kmc, normtrain)
clusterTrain = predict(km.kcca)
clusterTest = predict(km.kcca, newdata=normtest)
str(clusterTest)
table(clusterTest)

#Spliting train dataset into different clusters
splitTrain1 = subset(train, clusterTrain == 1)
splitTrain2 = subset(train, clusterTrain == 2)
splitTrain3 = subset(train, clusterTrain == 3)
splitTrain4 = subset(train, clusterTrain == 4)
splitTrain5 = subset(train, clusterTrain == 5)

#Spliting test dataset into different clusters
splitTest1 = subset(test, clusterTest == 1)
splitTest2 = subset(test, clusterTest == 2)
splitTest3 = subset(test, clusterTest == 3)
splitTest4 = subset(test, clusterTest == 4)
splitTest5 = subset(test, clusterTest == 5)

#Logistic regression models for each cluster
LogModel1=glm(Party~YOB+Gender+Income+HouseholdStatus+EducationLevel+Q124742+Q121699+Q121700+Q120194+Q118232+Q116197+Q115611+Q114517+Q113181+Q112478+Q111220+Q108950+Q109244+Q108342+Q102687+Q101596+Q100689+Q99716+Q98869+Q98578+Q98197,data=splitTrain1,family=binomial)
PredLog1=predict(LogModel1,splitTest1,type="response")
PredTestLabels1 = as.factor(ifelse(PredLog1<threshold, "Democrat", "Republican"))

LogModel2=glm(Party~YOB+Gender+Income+HouseholdStatus+EducationLevel+Q124742+Q121699+Q121700+Q120194+Q118232+Q116197+Q115611+Q114517+Q113181+Q112478+Q111220+Q108950+Q109244+Q108342+Q102687+Q101596+Q100689+Q99716+Q98869+Q98578+Q98197,data=splitTrain2,family=binomial)
PredLog2=predict(LogModel2,splitTest2,type="response")
PredTestLabels2 = as.factor(ifelse(PredLog2<threshold, "Democrat", "Republican"))

LogModel3=glm(Party~YOB+Gender+Income+HouseholdStatus+EducationLevel+Q124742+Q121699+Q121700+Q120194+Q118232+Q116197+Q115611+Q114517+Q113181+Q112478+Q111220+Q108950+Q109244+Q108342+Q102687+Q101596+Q100689+Q99716+Q98869+Q98578+Q98197,data=splitTrain3,family=binomial)
PredLog3=predict(LogModel3,splitTest3,type="response")
PredTestLabels3 = as.factor(ifelse(PredLog3<threshold, "Democrat", "Republican"))

LogModel4=glm(Party~YOB+Gender+Income+HouseholdStatus+EducationLevel+Q124742+Q121699+Q121700+Q120194+Q118232+Q116197+Q115611+Q114517+Q113181+Q112478+Q111220+Q108950+Q109244+Q108342+Q102687+Q101596+Q100689+Q99716+Q98869+Q98578+Q98197,data=splitTrain4,family=binomial)
PredLog4=predict(LogModel4,splitTest4,type="response")
PredTestLabels4 = as.factor(ifelse(PredLog4<threshold, "Democrat", "Republican"))

LogModel5=glm(Party~YOB+Gender+Income+HouseholdStatus+EducationLevel+Q124742+Q121699+Q121700+Q120194+Q118232+Q116197+Q115611+Q114517+Q113181+Q112478+Q111220+Q108950+Q109244+Q108342+Q102687+Q101596+Q100689+Q99716+Q98869+Q98578+Q98197,data=splitTrain5,family=binomial)
PredLog5=predict(LogModel5,splitTest5,type="response")
PredTestLabels5 = as.factor(ifelse(PredLog5<threshold, "Democrat", "Republican"))

#Combining the cluster predictions together
AllPredictions =as.factor(c(PredTestLabels1, PredTestLabels2, PredTestLabels3,PredTestLabels4,PredTestLabels5))
submitC=data.frame(USER_ID=test$USER_ID,Predictions=AllPredictions)
write.csv(submitC, file =  "cluster.csv", row.names = FALSE)

#Building a Lasso Regression model
library(glmnet)
x.train<-model.matrix(form,data=train)
x.test<-model.matrix(form,data=test2)
glmnet.mod<-glmnet(x.train,train$Party,alpha=1,family='binomial')
plot(glmnet.mod,xvar="lambda",xlab="log(Lambda)",ylab="Coefficients")
title("Coefficients path using Lasso Regression ",line=+3)
cv.lasso = cv.glmnet(x.train, train$Party, family = "binomial", alpha = 1)
plot(cv.lasso)
title("Deviance vs Lambda",line=+3)
best_lambda<-cv.lasso$lambda.min
predLasso = predict(cv.lasso, newx = x.test, s = "lambda.min")
predLassoLabels = as.factor(ifelse(predLasso < 0, "Democrat", "Republican"))
table(test2$Party,predLassoLabels)


test$Party = ""
x.submit = model.matrix(form, data = test)

predLasso = predict(cv.lasso, newx = x.submit, s = "lambda.min",type="response")
predLassoTrain=predict(cv.lasso,x.train,s="lambda.min",type="response")
predSubmitLabelsLasso = as.factor(ifelse(predLasso < 0.5, "Democrat", "Republican"))
submissionL = data.frame(USER_ID = test$USER_ID, Predictions = predSubmitLabelsLasso)
write.csv(submissionL, "glmnet_Lasso.csv", row.names = FALSE)


#Ridge Regression Model
ridge = glmnet(x.train, train$Party, family = "binomial", alpha = 0)
plot(ridge, xvar = "lambda", label = TRUE)
title("Coefficients Path using Ridge Regression",line=+3)
cv.ridge = cv.glmnet(x.train, train$Party, family = "binomial", alpha = 0)
plot(cv.ridge)
title("Deviance vs Lambda",line=+3)
predRidge = predict(cv.ridge, newx = x.test, s = "lambda.min")
predTestLabels = as.factor(ifelse(predRidge < 0, "Democrat", "Republican"))
table(predTestLabels,test2$Party)

predSubmitRidge = predict(cv.ridge, newx = x.submit, s = "lambda.min")
predSubmitLabelsRidge = as.factor(ifelse(predSubmitRidge < 0, "Democrat", "Republican"))
submissionR = data.frame(USER_ID = test$USER_ID, Predictions = predSubmitLabelsRidge)
write.csv(submissionR, "ridge_regression.csv", row.names = FALSE)

#Repeated Cross-Validation for out of sample accuracy or Ensemble Modelling
library(doParallel)
registerDoParallel(cores=4)

cv.folds <- 5
cv.repeats <- 3
tuneLength.set <- 5

library(caret)

set.seed(321)
seeds <- vector(mode = "list", length = (cv.folds*cv.repeats +1))
for(i in 1:(cv.folds*cv.repeats)) seeds[[i]] <- sample.int(100000, tuneLength.set)
seeds[[cv.folds*cv.repeats +1]] <- 456 

ctrl <- trainControl(method = "repeatedcv",number = cv.folds,repeats = cv.repeats,classProbs = TRUE,allowParallel = TRUE,savePredictions = "final",index=createResample(train$Party,25))

#Bagged Logistic Regression
set.seed(12345)
glm.mod<-train(form,data=train,family=binomial,method="glm",trControl=ctrl)
pGLM=predict(glm.mod,newdata=test,type="prob")[,2]
plabel=as.factor(ifelse(pGLM<threshold, "Democrat", "Republican"))
submitGLM=data.frame(USER_ID=test$USER_ID,Predictions=plabel)
write.csv(submitGLM, file =  "GLM2.csv", row.names = FALSE)

#Bagged CART
cv<-train(form,data=train,method="rpart",trControl=ctrl,tuneLength=tuneLength.set)
pCART=predict(cv,newdata=test,type="prob")[,2]
pCARTlabel=as.factor(ifelse(pCART<threshold, "Democrat", "Republican"))
submitbagCART=data.frame(USER_ID=test$USER_ID,Predictions=pCARTlabel)
write.csv(submitbagCART, file =  "baggedCART.csv", row.names = FALSE)

#Bagged GLMnet
glmnet.cv<-train(form,data=train,trControl=ctrl,method="glmnet",family="binomial",tuneLength=tuneLength.set)
pGLMnet=predict(glmnet.cv,newdata=test,type="prob")[,2]
pGLMlabel=as.factor(ifelse(pGLMnet<threshold, "Democrat", "Republican"))
submitGLMnet=data.frame(USER_ID=test$USER_ID,Predictions=pGLMlabel)
write.csv(submitGLMnet, file =  "trainedGLMnet.csv", row.names = FALSE)

bagging_results <- resamples(list(glm=glm.mod,CART=cv,glmnet=glmnet.cv))

summary(bagging_results)

#Ensemble-Boosting
#Gradient Boost Method model
GBMfit<-train(form,data=train,method="gbm",trControl=ctrl,verbose=FALSE)
GBMpred<-predict(GBMfit,newdata=test,type="prob")[,2]
PredTestGBM = as.factor(ifelse(GBMpred<threshold, "Democrat", "Republican"))
table(PredTestGBM,test2$Party)
submitGBM=data.frame(USER_ID=test$USER_ID,Predictions=PredTestGBM)
write.csv(submitGBM, file =  "GBM.csv", row.names = FALSE)

#C5.0 Gradient Boosting
C5fit<-train(form,data=train,method="C5.0",trControl=ctrl,verbose=FALSE,metric="Accuracy")
C5pred<-predict(C5fit,newdata=test,type="prob")[,2]
PredTestC5 = as.factor(ifelse(C5pred<threshold, "Democrat", "Republican"))
table(PredC5GBM,test2$Party)
submitC5=data.frame(USER_ID=test$USER_ID,Predictions=PredTestC5)
write.csv(submitC5, file =  "C5.csv", row.names = FALSE)



#ADA boost using Caret
library(fastAdaboost)
ADAfit2<-train(x=train[formvars],y=train$Party,method="ada",trControl=ctrl,verbose=FALSE,metric="Accuracy")
ADApred<-predict(ADAfit2,newdata=test,type="prob")[,2]
PredTrainADA<-predict(ADAfit2,type="prob")[,2]
PredTestADA = as.factor(ifelse(ADApred<threshold, "Democrat", "Republican"))
submitADA=data.frame(USER_ID=test$USER_ID,Predictions=PredTestADA)
write.csv(ADA, file =  "ada.csv", row.names = FALSE)

#ADA bagging using caret
ADAfit<-train(x=train[formvars],y=train$Party,method="AdaBag",trControl=ctrl,verbose=FALSE,metric="Accuracy")
predADA<-predict(ADAfit,newdata=test,type="prob")[,2]
PredADALabel = as.factor(ifelse(predADA<threshold, "Democrat", "Republican"))
table(test2$Party,PredADALabel)
submitADA=data.frame(USER_ID=test$USER_ID,Predictions=PredADALabel)
write.csv(submitADA, file =  "ADA.csv", row.names = FALSE)

Boosting_results <- resamples(list(gbm=GBMfit,C5=C5fit,ada=ADAfit2,Adabag=ADAfit))
summary(Boosting_results)

#XGBoost using Caret
library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)


xgfit<-train(x=X,y=train$Party,method="xgbTree",trControl=ctrl,verbose=FALSE)
predxg<-predict(xgfit,newdata=test,type="prob")[,2]
PredXGLabel = as.factor(ifelse(predxg<threshold, "Democrat", "Republican"))
table(test2$Party,PredXGLabel)
submitXG=data.frame(USER_ID=test$USER_ID,Predictions=PredXGLabel)
write.csv(submitXG, file =  "XGB.csv", row.names = FALSE)

#SVM Model
library(e1071)
library(caret)
library(kernlab)

#Validation set for SVM
dftrain2<-train2
dftest2<-test2
dftrain2$Party<-as.numeric(train2$Party=="Republican")
labels2<-dftrain2['Party']
dftrain2<-dftrain2[-grep('Party',colnames(dftrain2))]
dftest2<-dftest2[-grep('Party',colnames(dftest2))]
dfall2<-rbind(dftrain2,dftest2)

ohe_feats<-formvars
library(ade4)
library(data.table)

for (f in ohe_feats)
{
  df_all_dummy2 = acm.disjonctif(dfall2[f])
  dfall2[f] = NULL
  dfall2 = cbind(dfall2, df_all_dummy2)
}

newdfall2<-dfall2[,sapply(dfall2,is.numeric)]
X2=newdfall2[newdfall2$USER_ID%in%dftrain2$USER_ID,]
X2$YOB.2003<-NULL
y2<-labels2$Party
Xtest2<-newdfall2[newdfall2$USER_ID%in%dftest2$USER_ID,]
Xtest2$YOB.2003<-NULL

svm.model2<-svm(x=X2,y=y2,kernel="sigmoid",C=10,gamma=0.0006)
svm.predict2<-predict(svm.model2,newdata=Xtest2,type="response")
svm.predict.Label2 = as.factor(ifelse(svm.predict2<threshold, "Democrat", "Republican"))
table(test2$Party,svm.predict.Label2)

svm.model<-svm(x=X,y=y,kernel="radial",C=1,gamma=0.005)
svm.predict<-predict(svm.model,newdata=Xtest,type="response")
svm.predict.Label = as.factor(ifelse(svm.predict<threshold, "Democrat", "Republican"))
submitSVM=data.frame(USER_ID=test$USER_ID,Predictions=svm.predict.Label)
write.csv(submitSVM, file =  "svmRad.csv", row.names = FALSE)

svm.model.poly<-svm(x=X,y=y,kernel="polynomial",C=1,gamma=0.005)
svm.predict.poly<-predict(svm.model.poly,newdata=Xtest,type="response")
svm.predict.Label.p = as.factor(ifelse(svm.predict.poly<threshold, "Democrat", "Republican"))
submitSVMP=data.frame(USER_ID=test$USER_ID,Predictions=svm.predict.Label.p)
write.csv(submitSVMP, file =  "svmPoly.csv", row.names = FALSE)

svm.model.sig<-svm(x=X,y=y,kernel="sigmoid",C=1,gamma=0.0006)
svm.predict.s<-predict(svm.model.sig,newdata=Xtest,type="response")
svm.predict.Label.s = as.factor(ifelse(svm.predict.s<threshold, "Democrat", "Republican"))
submitSVMS=data.frame(USER_ID=test$USER_ID,Predictions=svm.predict.Label.s)
write.csv(submitSVMS, file =  "svmSig.csv", row.names = FALSE)



#XGBoost Algorithm
library(Matrix)
library(MatrixModels)
sparse_matrix <- sparse.model.matrix(Party ~ .-1, data = train)
dftrain<-train
dftest<-test
dftrain$Party<-as.numeric(train$Party=="Republican")
labels<-dftrain['Party']
dftrain<-dftrain[-grep('Party',colnames(dftrain))]
dftest<-dftest[-grep('Party',colnames(dftest))]
dfall<-rbind(dftrain,dftest)
ohe_feats<-formvars
library(ade4)
library(data.table)
for (f in ohe_feats)
  {
  df_all_dummy = acm.disjonctif(dfall[f])
  dfall[f] = NULL
  dfall = cbind(dfall, df_all_dummy)
}

newdfall<-dfall[,sapply(dfall,is.numeric)]
X=newdfall[newdfall$USER_ID%in%dftrain$USER_ID,]
y<-labels$Party
Xtest<-newdfall[newdfall$USER_ID%in%dftest$USER_ID,]

#Cross Validation for XGBoost
param = list("objective" = "binary:logistic","eta" = 0.0063, "max.depth" = 5 ,"nthread"=8,gamma=0,subsample=0.7,colsample_bytree=1,min_child_weight=1)
cv.nround = 800
bst.cv = xgb.cv(params=param,data = data.matrix(X[-1]),label = y,nfold = 5,eval_metric="auc",nrounds=cv.nround,prediction=T,verbose = TRUE,showsd=TRUE,stratified = TRUE,print,early.stop.round = 10)

#Plot the Test and Train auc means plot
library(dplyr)
library(tidyr)
bst.cv$dt %>%
  select(-contains("std")) %>%
  mutate(IterationNum = 1:n()) %>%
  gather(TestOrTrain, AUC, -IterationNum) %>%
  ggplot(aes(x = IterationNum, y = AUC, group = TestOrTrain, color = TestOrTrain)) + 
  geom_line() + 
  theme_bw()
bestRound<-which.max(bst.cv$dt$test.auc.mean-bst.cv$dt$test.auc.std)
xgb<-xgboost(data=data.matrix(X[-1]),label=y,params=param,verbose=TRUE,nrounds=bestRound)
predXGBtrain<-predict(xgb,data.matrix(X[-1]),ntreelimit=bestRound)
predXGB<-predict(xgb,data.matrix(Xtest[,-1]),ntreelimit=bestRound)

PredXGBLabel = as.factor(ifelse(predXGB<threshold, "Democrat", "Republican"))
submitXGB=data.frame(USER_ID=test$USER_ID,Predictions=PredXGBLabel)
write.csv(submitXGB, file =  "XGBoost.csv", row.names = FALSE)

#Training the XGB to find optimal parameters
xgb.grid <- expand.grid(nrounds = 800,
                        eta = seq(0.06,0.09,0.01),
                        max_depth = c(4,5,6),
                        gamma=0,
                        colsample_bytree=1,
                        min_child_weight=1
                        
)
xgb_trcontrol_1 = trainControl(
  method = "cv",
  number = 5,
  repeats=1,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "all",                                                        # save losses across all models
  classProbs = TRUE,                                                           # set to TRUE for AUC to be computed
  summaryFunction = twoClassSummary,
  allowParallel = TRUE
)

xgb_train=train(x = data.matrix(X[-1]),y= train$Party,trControl=xgb_trcontrol_1,tuneGrid = xgb.grid,method = "xgbTree",nthread=8)

#XGB model tree
model<-xgb.dump(xgb,with.stats = T)
model[1:10]
names<-dimnames( data.matrix(X[-1]))[[2]]
#Feature importance matrix
importance_matrix<-xgb.importance(names,model=xgb)
xgb.plot.importance(importance_matrix[41:50,])
hist(importance_matrix$Gain,xlim=c(0,0.05),xlab="Variable Gains",main="Histrogram of Variable Importance Gains")

modellist<-data.frame(list(predXGBtrain,PredTrainADA,predLassoTrain,PredRFTrain))
modellist<-matrix(modellist,nrow=4,ncol=5568,byrow=TRUE)
modellistTest=as.data.frame(list(predXGB,ADApred,predLasso,PredRF))
modellistTest<-matrix(modellistTest,nrow=4,ncol=1392,byrow=TRUE)
y<-matrix(y)
nn <- dbn.dnn.train(modellist,y,hidden = c(1),
                    activationfun = "sigm",learningrate = 0.2,momentum = 0.8)
nn_predict <- nn.predict(nn,modellist)
nn_predict_test <- nn.predict(nn,modellistTest)

#Greedy Ensemble
-
  


p<-as.data.frame(predict(model_list,newdata=head(test)))
modelCor(resamples(model_list))
results<-resamples(model_list)

model_preds <- lapply(model_list, predict, newdata=Xtest, type="prob")
model_preds <- data.frame(model_preds)
PredNNLabel = as.factor(ifelse(model_preds$nnet<threshold, "Democrat", "Republican"))
submitNN=data.frame(USER_ID=test$USER_ID,Predictions=PredNNLabel)
write.csv(submitNN, file =  "NN.csv", row.names = FALSE)

summary(results)

modelCor(results)

greedyensemble<-caretEnsemble(model_list,trControl=trainControl(number=10,classProbs = TRUE))
ens_preds <- predict(greedyensemble, newdata=Xtest, type="prob")
PredGELabel = as.factor(ifelse(ens_preds<threshold, "Democrat", "Republican"))
submitGE=data.frame(USER_ID=test$USER_ID,Predictions=PredGELabel)
write.csv(submitGE, file =  "greed_ens.csv", row.names = FALSE)


#GLM stack
glm_ensemble <- caretStack(
  model_list,
  method="bayesglm",
  metric="ROC",
  trControl=trainControl(
    method="boot",
    number=10,
    savePredictions="final",
    classProbs=TRUE)
)
predGLME<-predict(glm_ensemble,newdata=Xtest,type="prob")
PredGLMELabel = as.factor(ifelse(predGLME<threshold, "Democrat", "Republican"))
submitGLME=data.frame(USER_ID=test$USER_ID,Predictions=PredGLMELabel)
write.csv(submitGLME, file =  "glm_ens.csv", row.names = FALSE)

#GBM stack
gbm_ensemble <- caretStack(
  model_list,
  method="gbm",
  trControl=trainControl(
    method="boot",
    number=10,
    savePredictions="final",
    classProbs=TRUE)
)
predGBME<-predict(gbm_ensemble,newdata=Xtest,type="prob")
PredGBMELabel = as.factor(ifelse(predGBME<threshold, "Democrat", "Republican"))
submitGBME=data.frame(USER_ID=test$USER_ID,Predictions=PredGBMELabel)
write.csv(submitGBME, file =  "gbm_ens.csv", row.names = FALSE)


