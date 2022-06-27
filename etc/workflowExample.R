## Workflow example (Classification) ##

## 유저로부터 받는 입력은 camel case,
##예시로 사용한 변수 및 snake case로 작성된 dependencies의 함수명은 snake case로 표기합니다.


## data import

library(tidymodels)
library(dplyr)
library(recipes)
library(parsnip)
library(tune)
library(rsample)
library(vip)
library(goophi)

set.seed(1234)

## data import
data(titanic_train, package = "titanic")

cleaned_data <- tibble::as_tibble(titanic_train) %>%
  select(-c(PassengerId, Name, Cabin, Ticket)) %>%
  mutate(across(where(is.character), factor)) %>%
  mutate(Survived = as.factor(Survived ))

## one-hot encoding
rec <- recipe(Survived ~ ., data = cleaned_data) %>%
  step_dummy(all_predictors(), -all_numeric())

rec_prep <- prep(rec)

cleaned_data <- bake(rec_prep, new_data = cleaned_data)

## 여기까지 완료된 데이터가 전달된다고 가정 (one-hot encoding까지 되는지 확인 필요) ##


#### (1) Train-test split ####

# target 변수를 사용자로부터 입력 받습니다
targetVar <- "Survived"

# 아래 3 가지 data를 생성합니다.
data_train <- goophi::trainTestSplit(data = cleaned_data, target = targetVar)[[1]] # train data
data_test <- goophi::trainTestSplit(data = cleaned_data, target = targetVar)[[2]] # test data
data_split <- goophi::trainTestSplit(data = cleaned_data, target = targetVar)[[3]] # whole data with split information

#### (2) Make recipe for CV ####

# 아래 변수들을 사용자로부터 입력 받습니다
imputation <- TRUE
normalization <- TRUE
pca <- FALSE ## need to fix warning
formula <- "Survived ~ ."
pcaThres <- "0.7"

# train data에 대한 전처리 정보가 담긴 recipe를 생성합니다.
rec <- goophi::preprocessing(data = data_train,
                             formula,
                             imputationType = "mean",
                             normalizationType = "range", # min-max normalization as default
                             pcaThres = pcaThres,
                             imputation = imputation,
                             normalization = normalization,
                             pca = pca)
rec

#### (3) Modeling ####
## todo: make goophi to install dependencies when the engine is not installed

# engine, mode 사용자로부터 입력 받습니다
engine = "ranger"
mode = "classification"

# 사용자정의 ML 모델을 생성합니다
model <- goophi::randomForest_phi(trees = tune(),
                                  min_n = tune(),
                                  mtry = tune(),
                                  engine = engine,
                                  mode = mode)

model

#### (4) Grid serach CV ####

# 모델에 사용되는 parameter들을 사용해 parameterGrid를 입력받습니다 (사용자로부터 parameter grid를 받는 방법 고민)
parameterGrid <- dials::grid_regular(
  min_n(range = c(10, 40)),
  mtry(range = c(1, 5)),
  trees(range = c(500, 2000)),
  levels = 5)
# trining data를 몇 개로 나눌지 입력받습니다.
v <- 2

# parameter grid를 적용한 cross validation을 수행합니다
grid_search_result <- goophi::gridSerachCV(rec = rec,
                           model = model,
                           v = v,
                           data = data_train,
                           parameterGrid = parameterGrid
)
grid_search_result


#### (5) Finalize model ####

# 최종 모델 object를 생성합니다
finalized <- goophi::fitBestModel(gridSearchResult = grid_search_result,
                                  metric = "roc_auc",
                                  model = model,
                                  formula = formula,
                                  trainingData = data_train,
                                  splitedData = data_split)

final_model <- finalized[[1]]
last_fitted_model <- finalized[[2]]

final_model
last_fitted_model


## 아래 부분까지 문제가 없다면 함수화를 마무리합니다

last_fitted_model %>% collect_metrics()



