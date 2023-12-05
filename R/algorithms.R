#' Logistic Regression
#'
#' @description
#' The function for training user-defined Logistic regression model.
#'
#' This function supports: binary classification
#'
#' @details
#' Hyperparameters for tuning: penalty, mixture
#'
#' @param algo A name of the algorithm which can be customized by user (default: "Logistic Regression").
#' @param engine  The name of software that should be used to fit the model (Option: "glmnet" (default)).
#' @param mode  The model type. It should be "classification" or "regression" (Option: "classification" (default)).
#' @param trainingData The training data.
#' @param splitedData The whole dataset including the information of each fold
#' @param formula formula for modeling
#' @param rec Recipe object containing preprocessing information for cross-validation
#' @param v Applying v-fold cross validation in modeling process (default: 5)
#' @param gridNum Initial number of iterations to run before starting the optimization algorithm.
#' @param iter The maximum number of search iterations.
#' @param metric Metric to evaluate the performance (classification: "roc_auc" (default), "accuracy" / regression: "rmse" (default), "rsq").
#' @param seed Seed for reproducible results.
#'
#' @importFrom magrittr %>%
#' @importFrom dials penalty mixture
#' @import parsnip
#' @import glmnet
#'
#' @export

logisticRegression <- function(algo = "Logistic Regression",
                               engine = "glmnet",
                               mode = "classification",
                               trainingData = NULL,
                               splitedData = NULL,
                               formula = NULL,
                               rec = NULL,
                               v = 5,
                               gridNum = 5,
                               iter = 10,
                               metric = "roc_auc",
                               seed = 1234) {
  model <- parsnip::logistic_reg(
    penalty = tune(),
    mixture = tune()
  ) %>%
    parsnip::set_engine(engine = engine) %>%
    parsnip::set_mode(mode = mode) %>%
    parsnip::translate()

  bayes_opt_result <- stove::bayesOptCV(
    rec = rec,
    model = model,
    v = as.numeric(v),
    trainingData = trainingData,
    gridNum = gridNum,
    iter = iter,
    seed = seed
  )

  finalized <- stove::fitBestModel(
    optResult = bayes_opt_result,
    metric = metric,
    model = model,
    formula = formula,
    trainingData = trainingData,
    splitedData = splitedData,
    modelName = paste0(algo, "_", engine)
  )

  return(list(finalized = finalized, bayes_opt_result = bayes_opt_result))
}

#' Multinomial Regression
#'
#' @description
#' The function for training user-defined Multinomial regression model.
#'
#' This function supports: multinomial classification
#'
#' @details
#' Hyperparameters for tuning: penalty, mixture
#'
#' @param algo A name of the algorithm which can be customized by user (default: "Multinomial Regression").
#' @param engine  The name of software that should be used to fit the model (Option: "glmnet" (default)).
#' @param mode  The model type. It should be "classification" or "regression" (Option: "classification" (default)).
#' @param trainingData A data frame for training.
#' @param splitedData A data frame including metadata of split.
#' @param formula formula for modeling.
#' @param rec Recipe object containing preprocessing information for cross-validation.
#' @param v Applying v-fold cross validation in modeling process (default: 5).
#' @param gridNum Initial number of iterations to run before starting the optimization algorithm.
#' @param iter The maximum number of search iterations.
#' @param metric Metric to evaluate the performance (classification: "roc_auc" (default), "accuracy" / regression: "rmse" (default), "rsq").
#' @param seed Seed for reproducible results.
#'
#' @importFrom magrittr %>%
#' @importFrom dials penalty mixture
#' @import parsnip
#' @import glmnet
#'
#' @export

multinomialRegression <- function(algo = "Multinomial Regression",
                                  engine = "glmnet",
                                  mode = "classification",
                                  trainingData = NULL,
                                  splitedData = NULL,
                                  formula = NULL,
                                  rec = NULL,
                                  v = 5,
                                  gridNum = 5,
                                  iter = 10,
                                  metric = "roc_auc",
                                  seed = 1234) {
  model <- parsnip::multinom_reg(
    penalty = tune(),
    mixture = tune()
  ) %>%
    parsnip::set_engine(engine = engine) %>%
    parsnip::set_mode(mode = mode) %>%
    parsnip::translate()

  bayes_opt_result <- stove::bayesOptCV(
    rec = rec,
    model = model,
    v = as.numeric(v),
    trainingData = trainingData,
    gridNum = gridNum,
    iter = iter,
    seed = seed
  )

  finalized <- stove::fitBestModel(
    optResult = bayes_opt_result,
    metric = metric,
    model = model,
    formula = formula,
    trainingData = trainingData,
    splitedData = splitedData,
    modelName = paste0(algo, "_", engine)
  )

  return(list(finalized = finalized, bayes_opt_result = bayes_opt_result))
}

#' Linear Regression
#'
#' @details
#' The function for training user-defined Linear Regression model.
#'
#' Hyperparameters for tuning: penalty, mixture
#'
#' @param algo A name of the algorithm which can be customized by user (default: "Linear Regression").
#' @param engine  The name of software that should be used to fit the model ("glmnet" (default), "lm", "glm", "stan").
#' @param mode  The model type. It should be "classification" or "regression" ("regression" (default)).
#' @param trainingData The training data.
#' @param splitedData A data frame including metadata of split.
#' @param formula formula for modeling
#' @param rec Recipe object containing preprocessing information for cross-validation.
#' @param v Applying v-fold cross validation in modeling process (default: 5).
#' @param gridNum Initial number of iterations to run before starting the optimization algorithm.
#' @param iter The maximum number of search iterations.
#' @param metric Metric to evaluate the performance (classification: "roc_auc" (default), "accuracy" / regression: "rmse" (default), "rsq").
#' @param seed Seed for reproducible results.
#'
#' @importFrom magrittr %>%
#' @importFrom dials penalty mixture
#' @import parsnip
#' @import stats glmnet rstanarm
#'
#' @export

linearRegression <- function(algo = "Linear Regression",
                             engine = "glmnet",
                             mode = "regression",
                             trainingData = NULL,
                             splitedData = NULL,
                             formula = NULL,
                             rec = NULL,
                             v = 5,
                             gridNum = 5,
                             iter = 10,
                             metric = "rmse",
                             seed = 1234) {
  model <- parsnip::linear_reg(
    penalty = tune(),
    mixture = tune()
  ) %>%
    parsnip::set_engine(engine = engine) %>%
    parsnip::set_mode(mode = mode) %>%
    parsnip::translate()

  bayes_opt_result <- stove::bayesOptCV(
    rec = rec,
    model = model,
    v = as.numeric(v),
    trainingData = trainingData,
    gridNum = gridNum,
    iter = iter,
    seed = seed
  )

  finalized <- stove::fitBestModel(
    optResult = bayes_opt_result,
    metric = metric,
    model = model,
    formula = formula,
    trainingData = trainingData,
    splitedData = splitedData,
    modelName = paste0(algo, "_", engine)
  )


  return(list(finalized = finalized, bayes_opt_result = bayes_opt_result))
}


#' K-Nearest Neighbors
#'
#' @details
#' The function for training user-defined K-Nearest Neighbors model.
#'
#' Hyperparameters for tuning: neighbors
#'
#' @param algo A name of the algorithm which can be customized by user (default: "KNN").
#' @param engine  The name of software that should be used to fit the model ("kknn" (default)).
#' @param mode  The model type. It should be "classification" or "regression" ("classification" (default), "regression").
#' @param trainingData The training data.
#' @param splitedData A data frame including metadata of split.
#' @param formula formula for modeling
#' @param rec Recipe object containing preprocessing information for cross-validation.
#' @param v Applying v-fold cross validation in modeling process (default: 5).
#' @param gridNum Initial number of iterations to run before starting the optimization algorithm.
#' @param iter The maximum number of search iterations.
#' @param metric Metric to evaluate the performance (classification: "roc_auc" (default), "accuracy" / regression: "rmse" (default), "rsq").
#' @param seed Seed for reproducible results.
#'
#' @importFrom magrittr %>%
#' @importFrom dials neighbors
#' @import parsnip
#' @import kknn
#'
#' @export

KNN <- function(algo = "KNN",
                engine = "kknn",
                mode = "classification",
                trainingData = NULL,
                splitedData = NULL,
                formula = NULL,
                rec = NULL,
                v = 5,
                gridNum = 5,
                iter = 10,
                metric = NULL,
                seed = 1234) {
  model <- parsnip::nearest_neighbor(neighbors = tune()) %>%
    parsnip::set_engine(engine = engine) %>%
    parsnip::set_mode(mode = mode) %>%
    parsnip::translate()

  bayes_opt_result <- stove::bayesOptCV(
    rec = rec,
    model = model,
    v = as.numeric(v),
    trainingData = trainingData,
    gridNum = gridNum,
    iter = iter,
    seed = seed
  )

  finalized <- stove::fitBestModel(
    optResult = bayes_opt_result,
    metric = metric,
    model = model,
    formula = formula,
    trainingData = trainingData,
    splitedData = splitedData,
    modelName = paste0(algo, "_", engine)
  )

  return(list(finalized = finalized, bayes_opt_result = bayes_opt_result))
}

#' Naive Bayes
#'
#' @details
#' The function for training user-defined Naive Bayes model.
#'
#' Hyperparameters for tuning: smoothness, Laplace
#'
#' @param algo A name of the algorithm which can be customized by user (default: "Naive Bayes").
#' @param engine  The name of software that should be used to fit the model ("klaR" (default), naivebayes).
#' @param mode  The model type. It should be "classification" or "regression" ("classification" (default)).
#' @param trainingData The training data.
#' @param splitedData A data frame including metadata of split.
#' @param formula formula for modeling
#' @param rec Recipe object containing preprocessing information for cross-validation.
#' @param v Applying v-fold cross validation in modeling process (default: 5).
#' @param gridNum Initial number of iterations to run before starting the optimization algorithm.
#' @param iter The maximum number of search iterations.
#' @param metric Metric to evaluate the performance (classification: "roc_auc" (default), "accuracy" / regression: "rmse" (default), "rsq").
#' @param seed Seed for reproducible results.
#'
#' @importFrom magrittr %>%
#' @import parsnip
#' @importFrom dials Laplace
#' @importFrom discrim smoothness
#' @import klaR naivebayes
#'
#' @export

naiveBayes <- function(algo = "Naive Bayes",
                       engine = "klaR",
                       mode = "classification",
                       trainingData = NULL,
                       splitedData = NULL,
                       formula = NULL,
                       rec = NULL,
                       v = 5,
                       gridNum = 5,
                       iter = 10,
                       metric = NULL,
                       seed = 1234) {
  model <- parsnip::naive_Bayes(
    smoothness = tune(),
    Laplace = tune()
  ) %>%
    parsnip::set_engine(engine = engine) %>%
    parsnip::set_mode(mode = mode) %>%
    parsnip::translate()

  bayes_opt_result <- stove::bayesOptCV(
    rec = rec,
    model = model,
    v = as.numeric(v),
    trainingData = trainingData,
    gridNum = gridNum,
    iter = iter,
    seed = seed
  )

  finalized <- stove::fitBestModel(
    optResult = bayes_opt_result,
    metric = metric,
    model = model,
    formula = formula,
    trainingData = trainingData,
    splitedData = splitedData,
    modelName = paste0(algo, "_", engine)
  )

  return(list(finalized = finalized, bayes_opt_result = bayes_opt_result))
}

#' Decision Tree
#'
#' @details
#' The function for training user-defined Decision Tree model.
#'
#' Hyperparameters for tuning: tree_depth, min_n, cost_complexity
#'
#' @param algo A name of the algorithm which can be customized by user (default: "Decision Tree").
#' @param engine  The name of software that should be used to fit the model ("rpart" (default), "C50", "partykit").
#' @param mode  The model type. It should be "classification" or "regression" ("classification" (default), "regression").
#' @param trainingData The training data.
#' @param splitedData A data frame including metadata of split.
#' @param formula formula for modeling
#' @param rec Recipe object containing preprocessing information for cross-validation.
#' @param v Applying v-fold cross validation in modeling process (default: 5).
#' @param gridNum Initial number of iterations to run before starting the optimization algorithm.
#' @param iter The maximum number of search iterations.
#' @param metric Metric to evaluate the performance (classification: "roc_auc" (default), "accuracy" / regression: "rmse" (default), "rsq").
#' @param seed Seed for reproducible results.
#'
#' @importFrom magrittr %>%
#' @importFrom dials tree_depth min_n cost_complexity
#' @import parsnip
#' @import rpart C50 partykit
#'
#' @export

decisionTree <- function(algo = "Decision Tree",
                         engine = "rpart",
                         mode = "classification",
                         trainingData = NULL,
                         splitedData = NULL,
                         formula = NULL,
                         rec = NULL,
                         v = 5,
                         gridNum = 5,
                         iter = 10,
                         metric = NULL,
                         seed = 1234) {
  if (engine == "rpart") {
    model <- parsnip::decision_tree(
      cost_complexity = tune(),
      tree_depth = tune(),
      min_n = tune()
    ) %>%
      parsnip::set_engine(engine = engine) %>%
      parsnip::set_mode(mode = mode) %>%
      parsnip::translate()

    bayes_opt_result <- stove::bayesOptCV(
      rec = rec,
      model = model,
      v = as.numeric(v),
      trainingData = trainingData,
      gridNum = gridNum,
      iter = iter,
      seed = seed
    )

    finalized <- stove::fitBestModel(
      optResult = bayes_opt_result,
      metric = metric,
      model = model,
      formula = formula,
      trainingData = trainingData,
      splitedData = splitedData,
      modelName = paste0(algo, "_", engine)
    )
  } else if (engine == "C5.0") {
    model <- parsnip::decision_tree(
      min_n = tune()
    ) %>%
      parsnip::set_engine(engine = engine) %>%
      parsnip::set_mode(mode = mode) %>%
      parsnip::translate()

    bayes_opt_result <- stove::bayesOptCV(
      rec = rec,
      model = model,
      v = as.numeric(v),
      trainingData = trainingData,
      gridNum = gridNum,
      iter = iter,
      seed = seed
    )

    finalized <- stove::fitBestModel(
      optResult = bayes_opt_result,
      metric = metric,
      model = model,
      formula = formula,
      trainingData = trainingData,
      splitedData = splitedData,
      modelName = paste0(algo, "_", engine)
    )
  } else { # partykit

    model <- parsnip::decision_tree(
      tree_depth = tune(),
      min_n = tune()
    ) %>%
      parsnip::set_engine(engine = engine) %>%
      parsnip::set_mode(mode = mode) %>%
      parsnip::translate()

    bayes_opt_result <- stove::bayesOptCV(
      rec = rec,
      model = model,
      v = as.numeric(v),
      trainingData = trainingData,
      gridNum = gridNum,
      iter = iter,
      seed = seed
    )

    finalized <- stove::fitBestModel(
      optResult = bayes_opt_result,
      metric = metric,
      model = model,
      formula = formula,
      trainingData = trainingData,
      splitedData = splitedData,
      modelName = paste0(algo, "_", engine)
    )
  }

  return(list(finalized = finalized, bayes_opt_result = bayes_opt_result))
}


#' Random Forest
#'
#' @details
#' The function for training user-defined Random Forest model.
#'
#' Hyperparameters for tuning: trees, min_n, mtry
#'
#' @param algo A name of the algorithm which can be customized by user (default: "Random Forest").
#' @param engine  The name of software that should be used to fit the model ("rpart" (default), "randomForest", "partykit").
#' @param mode  The model type. It should be "classification" or "regression" ("classification" (default), "regression").
#' @param trainingData The training data.
#' @param splitedData A data frame including metadata of split.
#' @param formula formula for modeling
#' @param rec Recipe object containing preprocessing information for cross-validation.
#' @param v Applying v-fold cross validation in modeling process (default: 5).
#' @param gridNum Initial number of iterations to run before starting the optimization algorithm.
#' @param iter The maximum number of search iterations.
#' @param metric Metric to evaluate the performance (classification: "roc_auc" (default), "accuracy" / regression: "rmse" (default), "rsq").
#' @param seed Seed for reproducible results.
#'
#' @importFrom magrittr %>%
#' @importFrom dials mtry trees min_n
#' @import parsnip
#' @import ranger partykit
#'
#' @rawNamespace import(randomForest, except = c(margin, importance))
#'
#' @export

randomForest <- function(algo = "Random Forest",
                         engine = "ranger",
                         mode = "classification",
                         trainingData = NULL,
                         splitedData = NULL,
                         formula = NULL,
                         rec = NULL,
                         v = 5,
                         gridNum = 5,
                         iter = 10,
                         metric = NULL,
                         seed = 1234) {
  model <- parsnip::rand_forest(
    trees = tune(),
    min_n = tune(),
    mtry = tune()
  ) %>%
    parsnip::set_engine(engine = engine) %>%
    parsnip::set_mode(mode = mode) %>%
    parsnip::translate()

  bayes_opt_result <- stove::bayesOptCV(
    rec = rec,
    model = model,
    v = as.numeric(v),
    trainingData = trainingData,
    gridNum = gridNum,
    iter = iter,
    seed = seed
  )

  finalized <- stove::fitBestModel(
    optResult = bayes_opt_result,
    metric = metric,
    model = model,
    formula = formula,
    trainingData = trainingData,
    splitedData = splitedData,
    modelName = paste0(algo, "_", engine)
  )

  return(list(finalized = finalized, bayes_opt_result = bayes_opt_result))
}


#' XGBoost
#'
#' @description
#' The function for training user-defined XGBoost model.
#'
#' Hyperparameters for tuning: tree_depth, trees,learn_rate, mtry, min_n, loss_reduction, sample_size
#'
#' @details
#' XGBoost
#'
#' @param algo A name of the algorithm which can be customized by user (default: "XGBoost").
#' @param engine  The name of software that should be used to fit the model ("xgboost" (default)).
#' @param mode  The model type. It should be "classification" or "regression" ("classification" (default), "regression").
#' @param trainingData The training data.
#' @param splitedData A data frame including metadata of split.
#' @param formula formula for modeling
#' @param rec Recipe object containing preprocessing information for cross-validation.
#' @param v Applying v-fold cross validation in modeling process (default: 5).
#' @param gridNum Initial number of iterations to run before starting the optimization algorithm.
#' @param iter The maximum number of search iterations.
#' @param metric Metric to evaluate the performance (classification: "roc_auc" (default), "accuracy" / regression: "rmse" (default), "rsq").
#' @param seed Seed for reproducible results.
#'
#' @importFrom magrittr %>%
#' @importFrom dials tree_depth trees learn_rate mtry min_n loss_reduction sample_size stop_iter
#' @import parsnip treesnip
#'
#' @export

xgBoost <- function(algo = "XGBoost",
                    engine = "xgboost",
                    mode = "classification",
                    trainingData = NULL,
                    splitedData = NULL,
                    formula = NULL,
                    rec = NULL,
                    v = 5,
                    gridNum = 5,
                    iter = 10,
                    metric = NULL,
                    seed = 1234) {
  model <- parsnip::boost_tree(
    tree_depth = tune(),
    trees = tune(),
    learn_rate = tune(),
    mtry = tune(),
    min_n = tune(),
    loss_reduction = tune(),
    sample_size = tune()
  ) %>%
    parsnip::set_engine(engine = engine) %>%
    parsnip::set_mode(mode = mode) %>%
    parsnip::translate()

  bayes_opt_result <- stove::bayesOptCV(
    rec = rec,
    model = model,
    v = as.numeric(v),
    trainingData = trainingData,
    gridNum = gridNum,
    iter = iter,
    seed = seed
  )

  finalized <- stove::fitBestModel(
    optResult = bayes_opt_result,
    metric = metric,
    model = model,
    formula = formula,
    trainingData = trainingData,
    splitedData = splitedData,
    modelName = paste0(algo, "_", engine)
  )

  return(list(finalized = finalized, bayes_opt_result = bayes_opt_result))
}

#' Light GBM
#'
#' @details
#' The function for training user-defined Light GBM model.
#'
#' Hyperparameters for tuning: tree_depth, trees, learn_rate, mtry, min_n, loss_reduction
#'
#' @param algo A name of the algorithm which can be customized by user. (default: "lightGBM").
#' @param engine  The name of software that should be used to fit the model("lightgbm" (default)).
#' @param mode  The model type. It should be "classification" or "regression" ("classification" (default), "regression").
#' @param trainingData The training data.
#' @param splitedData A data frame including metadata of split.
#' @param formula formula for modeling
#' @param rec Recipe object containing preprocessing information for cross-validation.
#' @param v Applying v-fold cross validation in modeling process (default: 5).
#' @param gridNum Initial number of iterations to run before starting the optimization algorithm.
#' @param iter The maximum number of search iterations.
#' @param metric Metric to evaluate the performance (classification: "roc_auc" (default), "accuracy" / regression: "rmse" (default), "rsq").
#' @param seed Seed for reproducible results.
#'
#' @importFrom magrittr %>%
#' @importFrom dials tree_depth trees learn_rate mtry min_n loss_reduction
#' @import parsnip treesnip
#'
#' @export

lightGbm <- function(algo = "lightGBM",
                     engine = "lightgbm",
                     mode = "classification",
                     trainingData = NULL,
                     splitedData = NULL,
                     formula = NULL,
                     rec = NULL,
                     v = 5,
                     gridNum = 5,
                     iter = 15,
                     metric = NULL,
                     seed = 1234) {
  model <- parsnip::boost_tree(
    tree_depth = tune(),
    trees = tune(),
    learn_rate = tune(),
    mtry = tune(),
    min_n = tune(),
    loss_reduction = tune()
  ) %>%
    parsnip::set_engine(engine = engine, force_row_wise = T) %>%
    parsnip::set_mode(mode = mode) %>%
    parsnip::translate()

  bayes_opt_result <- stove::bayesOptCV(
    rec = rec,
    model = model,
    v = as.numeric(v),
    trainingData = trainingData,
    gridNum = gridNum,
    iter = iter,
    seed = seed
  )

  finalized <- stove::fitBestModel(
    optResult = bayes_opt_result,
    metric = metric,
    model = model,
    formula = formula,
    trainingData = trainingData,
    splitedData = splitedData,
    modelName = paste0(algo, "_", engine)
  )

  return(list(finalized = finalized, bayes_opt_result = bayes_opt_result))
}

#' SVMLinear
#'
#' @details
#' The function for training user-defined SVM Linear model.
#'
#' @param algo A name of the algorithm which can be customized by user (default: "SVMLinear").
#' @param engine  The name of software that should be used to fit the model ("kernlab" (default)).
#' @param mode  The model type. It should be "classification" or "regression" ("classification" (default), "regression").
#' @param trainingData The training data.
#' @param splitedData A data frame including metadata of split.
#' @param formula formula for modeling
#' @param rec Recipe object containing preprocessing information for cross-validation.
#' @param v Applying v-fold cross validation in modeling process (default: 5).
#' @param gridNum Initial number of iterations to run before starting the optimization algorithm.
#' @param iter The maximum number of search iterations.
#' @param metric Metric to evaluate the performance (classification: "roc_auc" (default), "accuracy" / regression: "rmse" (default), "rsq").
#' @param seed Seed for reproducible results.
#'
#' @importFrom magrittr %>%
#' @importFrom dials cost svm_margin
#' @import parsnip
#' @import kernlab
#'
#' @export

SVMLinear <- function(algo = "SVMLinear",
                      engine = "kernlab",
                      mode = "classification",
                      trainingData = NULL,
                      splitedData = NULL,
                      formula = NULL,
                      rec = NULL,
                      v = 5,
                      gridNum = 5,
                      iter = 15,
                      metric = NULL,
                      seed = 1234) {
  model <- parsnip::svm_linear(
    cost = tune(),
    margin = tune()
  ) %>%
    parsnip::set_engine(engine = engine) %>%
    parsnip::set_mode(mode = mode) %>%
    parsnip::translate()

  bayes_opt_result <- stove::bayesOptCV(
    rec = rec,
    model = model,
    v = as.numeric(v),
    trainingData = trainingData,
    gridNum = gridNum,
    iter = iter,
    seed = seed
  )

  finalized <- stove::fitBestModel(
    optResult = bayes_opt_result,
    metric = metric,
    model = model,
    formula = formula,
    trainingData = trainingData,
    splitedData = splitedData,
    modelName = paste0(algo, "_", engine)
  )

  return(list(finalized = finalized, bayes_opt_result = bayes_opt_result))
}

#' SVMPoly
#'
#' @details
#' The function for training user-defined SVM Poly model.
#'
#' @param algo A name of the algorithm which can be customized by user (default: "SVMPoly").
#' @param engine  The name of software that should be used to fit the model ("kernlab" (default)).
#' @param mode  The model type. It should be "classification" or "regression" ("classification" (default), "regression").
#' @param trainingData The training data.
#' @param splitedData A data frame including metadata of split.
#' @param formula formula for modeling
#' @param rec Recipe object containing preprocessing information for cross-validation.
#' @param v Applying v-fold cross validation in modeling process (default: 5).
#' @param gridNum Initial number of iterations to run before starting the optimization algorithm.
#' @param iter The maximum number of search iterations.
#' @param metric Metric to evaluate the performance (classification: "roc_auc" (default), "accuracy" / regression: "rmse" (default), "rsq").
#' @param seed Seed for reproducible results.
#'
#' @importFrom magrittr %>%
#' @importFrom dials cost degree scale_factor svm_margin
#' @import parsnip
#' @import kernlab
#'
#' @export

SVMPoly <- function(algo = "SVMPoly",
                    engine = "kernlab",
                    mode = "classification",
                    trainingData = NULL,
                    splitedData = NULL,
                    formula = NULL,
                    rec = NULL,
                    v = 5,
                    gridNum = 5,
                    iter = 15,
                    metric = NULL,
                    seed = 1234) {
  model <- parsnip::svm_poly(
    cost = tune(),
    degree = tune(),
    scale_factor = tune(),
    margin = tune()
  ) %>%
    parsnip::set_engine(engine = engine) %>%
    parsnip::set_mode(mode = mode) %>%
    parsnip::translate()

  bayes_opt_result <- stove::bayesOptCV(
    rec = rec,
    model = model,
    v = as.numeric(v),
    trainingData = trainingData,
    gridNum = gridNum,
    iter = iter,
    seed = seed
  )

  finalized <- stove::fitBestModel(
    optResult = bayes_opt_result,
    metric = metric,
    model = model,
    formula = formula,
    trainingData = trainingData,
    splitedData = splitedData,
    modelName = paste0(algo, "_", engine)
  )

  return(list(finalized = finalized, bayes_opt_result = bayes_opt_result))
}

#' SVMRbf
#'
#' @details
#' The function for training user-defined SVM Rbf model.
#'
#' @param algo A name of the algorithm which can be customized by user (default: "SVMRbf").
#' @param engine  The name of software that should be used to fit the model ("kernlab" (default)).
#' @param mode  The model type. It should be "classification" or "regression" ("classification" (default), "regression").
#' @param trainingData The training data.
#' @param splitedData A data frame including metadata of split.
#' @param formula formula for modeling
#' @param rec Recipe object containing preprocessing information for cross-validation.
#' @param v Applying v-fold cross validation in modeling process (default: 5).
#' @param gridNum Initial number of iterations to run before starting the optimization algorithm.
#' @param iter The maximum number of search iterations.
#' @param metric Metric to evaluate the performance (classification: "roc_auc" (default), "accuracy" / regression: "rmse" (default), "rsq").
#' @param seed Seed for reproducible results.
#'
#' @importFrom magrittr %>%
#' @importFrom dials cost rbf_sigma svm_margin
#' @import parsnip
#' @import kernlab
#'
#' @export

SVMRbf <- function(algo = "SVMRbf",
                   engine = "kernlab",
                   mode = "classification",
                   trainingData = NULL,
                   splitedData = NULL,
                   formula = NULL,
                   rec = NULL,
                   v = 5,
                   gridNum = 5,
                   iter = 15,
                   metric = NULL,
                   seed = 1234) {
  model <- parsnip::svm_rbf(
    cost = tune(),
    rbf_sigma = tune(),
    margin = tune()
  ) %>%
    parsnip::set_engine(engine = engine) %>%
    parsnip::set_mode(mode = mode) %>%
    parsnip::translate()

  bayes_opt_result <- stove::bayesOptCV(
    rec = rec,
    model = model,
    v = as.numeric(v),
    trainingData = trainingData,
    gridNum = gridNum,
    iter = iter,
    seed = seed
  )

  finalized <- stove::fitBestModel(
    optResult = bayes_opt_result,
    metric = metric,
    model = model,
    formula = formula,
    trainingData = trainingData,
    splitedData = splitedData,
    modelName = paste0(algo, "_", engine)
  )

  return(list(finalized = finalized, bayes_opt_result = bayes_opt_result))
}

#' neural network
#'
#' @details
#' The function for training user-defined MLP model.
#'
#' Hyperparameters for tuning: hidden_units, penalty, epochs
#'
#' @param algo A name of the algorithm which can be customized by user (default: "MLP").
#' @param engine  The name of software that should be used to fit the model ("nnet" (default)).
#' @param mode  The model type. It should be "classification" or "regression" ("classification" (default), "regression").
#' @param trainingData The training data.
#' @param splitedData A data frame including metadata of split.
#' @param formula formula for modeling
#' @param rec Recipe object containing preprocessing information for cross-validation.
#' @param v Applying v-fold cross validation in modeling process (default: 5).
#' @param gridNum Initial number of iterations to run before starting the optimization algorithm.
#' @param iter The maximum number of search iterations.
#' @param metric Metric to evaluate the performance (classification: "roc_auc" (default), "accuracy" / regression: "rmse" (default), "rsq").
#' @param seed Seed for reproducible results.
#'
#' @importFrom magrittr %>%
#' @importFrom dials hidden_units penalty epochs
#' @import parsnip
#'
#' @export

MLP <- function(algo = "MLP",
                engine = "nnet",
                mode = "classification",
                trainingData = NULL,
                splitedData = NULL,
                formula = NULL,
                rec = NULL,
                v = 5,
                gridNum = 5,
                iter = 10,
                metric = NULL,
                seed = 1234) {
  model <- parsnip::mlp(
    hidden_units = tune(),
    penalty = tune(),
    epochs = tune()
  ) %>%
    parsnip::set_engine(engine = engine) %>%
    parsnip::set_mode(mode = mode) %>%
    parsnip::translate()

  bayes_opt_result <- stove::bayesOptCV(
    rec = rec,
    model = model,
    v = as.numeric(v),
    trainingData = trainingData,
    gridNum = gridNum,
    iter = iter,
    seed = seed
  )

  finalized <- stove::fitBestModel(
    optResult = bayes_opt_result,
    metric = metric,
    model = model,
    formula = formula,
    trainingData = trainingData,
    splitedData = splitedData,
    modelName = paste0(algo, "_", engine)
  )

  return(list(finalized = finalized, bayes_opt_result = bayes_opt_result))
}

#' K means clustering
#'
#' @details
#' The function for K means clustering.
#'
#' parameters for tuning: maxK, nstart
#'
#' @param data 전처리가 완료된 데이터
#' @param maxK 클러스터링 수행 시 군집을 2, 3, ..., maxK개로 분할 (default: 15)
#' @param nstart 랜덤 샘플에 대해 초기 클러스터링을 nstart번 시행 (default: 25)
#' @param iterMax 반복계산을 수행할 최대 횟수 (default: 10)
#' @param nBoot gap statictic을 사용해 클러스터링을 수행할 때 Monte Carlo (bootstrap) 샘플의 개수 (selectOptimal == "gap_stat" 일 경우에만 지정, default: 100)
#' @param algorithm K means를 수행할 알고리즘 선택 ("Hartigan-Wong" (default), "Lloyd", "Forgy", "MacQueen")
#' @param selectOptimal 최적의 K값을 선정할 때 사용할 method 선택 ("silhouette" (default), "gap_stat")
#' @param seedNum seed값 설정
#'
#' @importFrom magrittr %>%


#' @import stats
#' @import factoextra
#'
#' @export

kMeansClustering <- function(data,
                             maxK = 15,
                             nStart = 25,
                             iterMax = 10,
                             nBoot = 100,
                             algorithm = "Hartigan-Wong", ## "Hartigan-Wong", "Lloyd", "Forgy", "MacQueen"
                             selectOptimal = "silhouette", # silhouette, gap_stat
                             seedNum = 6471) {
  colors <- grDevices::colorRampPalette(c("#C70A80", "#FBCB0A", "#3EC70B", "#590696", "#37E2D5"))
  set.seed(as.numeric(seedNum))

  elbowPlot <- factoextra::fviz_nbclust(
    x = data,
    FUNcluster = stats::kmeans,
    method = "wss"
  )

  if (selectOptimal == "silhouette") {
    result_clust <- factoextra::fviz_nbclust(
      x = data,
      FUNcluster = stats::kmeans,
      method = selectOptimal,
      k.max = as.numeric(maxK),
      barfill = "slateblue",
      barcolor = "slateblue",
      linecolor = "slateblue"
    )
    cols <- colors(result_clust$data$clusters[which.max(result_clust$data$y)])
    optimalK <- as.numeric(result_clust$data$clusters[which.max(result_clust$data$y)])
  } else if (selectOptimal == "gap_stat") {
    result_clust <- factoextra::fviz_nbclust(
      x = data,
      FUNcluster = stats::kmeans,
      method = selectOptimal,
      k.max = as.numeric(maxK),
      nboot = as.numeric(nBoot),
      barfill = "slateblue",
      barcolor = "slateblue",
      linecolor = "slateblue"
    )
    cols <- colors(result_clust$data$clusters[which.max(result_clust$data$gap)])

    optimalK <- as.numeric(result_clust$data$clusters[which.max(result_clust$data$gap)])
  } else {
    stop("selectOptimal must be 'silhouette' or 'gap_stat'.")
  }

  result <- stats::kmeans(
    x = data,
    centers = optimalK,
    iter.max = iterMax,
    nstart = as.numeric(nStart),
    algorithm = algorithm
  )

  clustVis <- factoextra::fviz_cluster(
    object = result,
    data = data,
    palette = cols,
    geom = "point",
    ellipse.type = "convex",
    ggtheme = theme_bw()
  )

  return(list(result = result, elbowPlot = elbowPlot, optimalK = result_clust, clustVis = clustVis))
}
