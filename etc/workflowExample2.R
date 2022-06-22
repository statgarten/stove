
## Classification model workflow ##

## data import

library(tidymodels)  # for the tune package, along with the rest of tidymodels
library(dplyr)
library(recipes)
library(parsnip)
library(tune)
library(rsample)

# install.packages("titanic")
data(titanic_train, package = "titanic")
cleaned_data <- tibble::as_tibble(titanic_train) %>%
  select(-PassengerId)

set.seed(1234) # fix seed

#### train-test split (goophi::trainTestSplit(data, strata)); tibble -> list(train, test) ####

## from
# data_split <- rsample::initial_split(cleaned_data, strata = Survived)
#
# data_train <- rsample::training(data_split)
# data_test  <- rsample::testing(data_split)

## to
targetVar <- "Survived"
data_train <- goophi::trainTestSplit(cleaned_data, strata = cleaned_data[[targetVar]])[[1]]
data_test <- goophi::trainTestSplit(cleaned_data, strata = cleaned_data[[targetVar]])[[2]]

nrow(data_train)
nrow(data_train)/nrow(cleaned_data)

data_train %>%
  count(Survived) %>%
  mutate(prop = n/sum(n))

data_test %>%
  count(Survived) %>%
  mutate(prop = n/sum(n))

#### define model goophi::selectModel( ... ); . -> list ####

## from
fitted_logistic_model<- parsnip::logistic_reg() %>%
  parsnip::set_engine("glm") %>%
  parsnip::set_mode("classification") %>%
  parsnip::fit(diabetes~., data = diabetes_train)

## to
f <- "targetVar~."
my_model <- goophi::selectModel()

#### Cross Validation (goophi::createCvObject(data, v, repeats)); tibble -> tibble ####
# create CV object from training data
v = 10
repeats = 1
data_train_cv <- rsample::vfold_cv(data = data_train, v = v, repeats = repeats)


#### imputation (goophi::createCvObject(data, v, repeats); list -> list) ####
userImputation = TRUE # if userImputation == TRUE, but any(is.na(cleaned_data)) == FALSE, do not perform imputation

if (userImputation == TRUE & any(is.na(cleaned_data))){
  rec <- recipe(Survived ~ ., data = data_train) %>% ## imputation as initial step of
    step_impute_bag() ## need to add options

  print("--------------Imputation has performed--------------")
}


#### Scaling ####
userScaling = TRUE

if (userScaling == TRUE){
  print("--------------Scaling has performed--------------")
}


#### PCA ####
userPca = TRUE

if (userPca == TRUE){
  print("--------------PCA has performed--------------")
}


lr_workflow <-
  workflow() %>%
  add_model(lr_mod) %>%
  add_recipe(lr_recipe)


#### set grid  + tune grid ? ####
my_grid <- expand.grid(disp = 2:5, wt = 2:5)

rec <-
  tune_grid(lr_mod, rec, resamples = data_train_cv, grid = my_grid)


#### model fitting ####
# Choose the best model obtained through grid search and perform fitting process
select_best(rec, metric = "rmse")
update_model()

finalModelObject <- last_fit()
