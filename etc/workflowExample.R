#### Workflow example ####

## data import

library(tidymodels)
library(dplyr)
library(recipes)
library(parsnip)
library(tune)
library(rsample)
library(vip)

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
## ##으로 구분된 파트를 묶어 함수화할 예정 ##

# train-test split (trainTestSplit(data, target))
targetVar <- "Survived"

data_train <- goophi::trainTestSplit(data = cleaned_data, target = targetVar)[[1]]
data_test <- goophi::trainTestSplit(data = cleaned_data, target = targetVar)[[2]]
data_split<-goophi::trainTestSplit(data = cleaned_data, target = targetVar)[[3]]

## make recipe for CV
pca_thres <- "0.7"
f <- "Survived~."

rec <- recipe(eval(parse(text = f)), data = data_train) %>%
  step_impute_knn(all_predictors()) %>% ## imputation
  step_center(all_predictors())  %>%
  step_scale(all_predictors()) %>% ## standardization or scaling
  step_pca(all_predictors(), threshold = eval(parse(text = pca_thres))) ## PCA for numeric var only or all predictors

## modeling
engine = "ranger"
mode = "classification"

# model <- parsnip::rand_forest(
#   mtry = tune(), # tune number of sampled predictors at each split
#   trees = tune(), # tune number of trees
#   min_n = tune()) %>% # tune minimum number of data points in a node
#   set_engine(engine = engine, importance = "impurity") %>% #engine-specific arguments
#   set_mode(mode = mode)

model <- randomForest_phi(formula = f,
                          trees = tune(), ## tune 넘길 때 null로 전달되는 문제
                          min_n = tune(),
                          mtry = tune(),
                          engine = engine,
                          mode = mode)

model

## set workflow
tune_wf <- workflow() %>%
  add_recipe(rec) %>%
  add_model(model)

## v-fold cv
folds <- vfold_cv(data_train, v = 2)

## parameter_grid / need to set default range
parameter_grid <- grid_regular(
  min_n(range = c(10, 40)),
  mtry(range = c(1, 5)),
  trees(range = c(500, 2000)),
  levels = 5)

## grid search CV
regular_res <- tune_grid(tune_wf, resamples = folds, grid = parameter_grid) # warnings


## results of grid search CV
regular_res %>% collect_metrics()
autoplot(regular_res)

## finalize model
show_best_params <- show_best(regular_res, n = 1, metric = "roc_auc") # regular_res$.metrics
Best_params <- select_best(regular_res, metric = "roc_auc")
final_spec <- finalize_model(model, Best_params)

final_model <- final_spec %>% fit(eval(parse(text = f)), data_train)

# last_fitted_model
last_fitted_model <-
  tune_wf %>%
  update_model(final_spec) %>%
  last_fit(data_split)

# performance of final model
last_fitted_model %>% collect_metrics()

# importance
last_fitted_model %>%
  extract_fit_parsnip() %>%
  vip(num_features = 5)

# ROC Curve
last_fitted_model %>%
  collect_predictions() %>%
  #mutate(Survived = as.numeric(Survived)) %>%
  #mutate(.pred_class = as.numeric(.pred_class)) %>%
  roc_curve(Survived, .pred_class) %>%
  autoplot()

# confusion matrix
last_fitted_model %>%
  collect_predictions() %>%
  conf_mat(Survived, .pred_class)


