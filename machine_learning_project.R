# Week 3 project

# Load libraries
library(caret)
library(doParallel)
registerDoParallel(cores=2)

# Useful function for wrtiting result files
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}


# Read in training and testing set
train <- read.csv(file="ml_training.csv")
test <- read.csv(file="ml_testing.csv")

# I now split the training set into a traintrain and a traintest set 
# (containing respectively 60% and 40% of the original data)
inTrain <- createDataPartition(y=train$classe, p=0.6, list=FALSE)
traintrain <- train[inTrain,]
traintest <- train[-inTrain,]

# The dataset has 160 columns - i.e. 159 predictors
# As a first step, I try to reduce this number
# First I look for variables with little variability and discard them
dummy <- nearZeroVar(traintrain, saveMetrics = TRUE)
newcols <- rownames(dummy[which(dummy$nzv==FALSE),])
train2 <- traintrain[, newcols]
# The first 6 column seem to be of no interest whatsoever for building the prediction model
train3 <- train2[,-c(1:6)]
# Several columns are made for over 97% of NaN values
# (These column all have NaN as a first entry)
# I am removing them
newcols2 <- colnames(train3[, which(!is.na(train3[1,]))])
train4 <- train3[,newcols2]
# Now remove the same columns from the traintest set
test4 <- traintest[,colnames(train4)]

## Let's take a subsample of the initial data
## train44 <- train4[sample(nrow(train4), 2000),]

## Case 1: direct training (eg random forest)
## print("Training begins...")
## t1 <- Sys.time()
## model1 <- train(classe ~., method="rf", data=train44)
## t2 <- Sys.time()
## print("...Finished!")
## print(t2-t1)
## modelTest1 <- predict(model1,test4)

# Case 2: run PCA first
preProc <- preProcess(train4[,-53], method="pca", thresh=0.90, na.remove=TRUE)
trainPC <- predict(preProc, train4[,-53])
# Training with Random Forests
print("Training begins...")
t1 <- Sys.time()
model2 <- train(train4$classe ~., method="rf", data=trainPC)
t2 <- Sys.time()
print("...Finished!")
print(t2-t1)

# Validation of the model (application to traintest set)
testPC <- predict(preProc, test4[,-53])
modelTest2 <- predict(model2,testPC)
## print(confusionMatrix(test4$classe, modelTest1))
## print("##############")
print(confusionMatrix(test4$classe, modelTest2))
dput(confusionMatrix(test4$classe, modelTest2), file='conf_Matrix.txt')

# Now to the unknown test set
# Remove unnecessary columns
cc <- colnames(test4)
test <- test[,cc[-53]]

## modelResults1 <- predict(model1, test)
## answers <- as.character(modelResults1)

# Apply the model to the test set
testPC <- predict(preProc, test)
modelResult2 <- predict(model2, testPC)

# Manipulate the results for submission
answers2 <- as.character(modelResult2)
## answers <-paste(answers,answers2)
print(answers2)
pml_write_files(answers2)

## plot <- ggplot(trainPC, aes(PC1, PC2, color=train4$classe)) +
##  geom_point()
## print(plot)




