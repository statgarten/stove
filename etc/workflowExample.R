#install.packages("mlr")
#install.packages("tidyverse")
# library(dplyr)
# library(mlr)
# library(tidyverse)
# library(phiml)

# data 불러오기
#install.packages("titanic")
data(titanic_train, package = "titanic")
titanicTib <- tibble::as_tibble(titanic_train)

# 클리닝
fctrs <- c("Survived", "Sex", "Pclass")
titanicClean <- titanicTib %>%
  mutate_at(.vars = fctrs, .funs = factor) %>%
  mutate(FamSize = SibSp + Parch) %>%
  select(Survived, Pclass, Sex, Age, Fare, FamSize)

# na -> 평균값 imputation
imp <- mlr::impute(titanicClean, cols = list(Age = imputeMean()))
sum(is.na(imp$data$Age))

# 모델 훈련 (https://mlr.mlr-org.com/articles/tutorial/integrated_learners.html)
# titanicTask <- mlr::makeClassifTask(data = imp$data, target = "Survived")
# logReg <- mlr::makeLearner("classif.logreg", predict.type = "prob")
# logRegModel <- mlr::train(logReg, titanicTask)


logRegModel <- phiml::LogisticRegression(data = imp$data, target = "Survived")$logRegModel
titanicTask <- phiml::LogisticRegression(data = imp$data, target = "Survived")$task

# logReg가 imputation을 포함하도록 세팅
logRegWrapper <- mlr::makeImputeWrapper("classif.logreg",
                                   cols = list(Age = imputeMean()))

# 10 fold CV를 이용해 학습
kFold <- mlr::makeResampleDesc(method = "RepCV", folds = 10, reps = 50,
                          stratify = TRUE)

logRegwithImpute <- mlr::resample(logRegWrapper, titanicTask,
                             resampling = kFold,
                             measures = list(acc, fpr, fnr))
logRegwithImpute

# coef
logRegModelData <- getLearnerModel(logRegModel)
stats::coef(logRegModelData)

# odds ratio
exp(cbind(Odds_Ratio = coef(logRegModelData), confint(logRegModelData)))


