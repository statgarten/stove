#' Bayesian optimization with cross validation
#'
#' @details
#' Optimize the hyperparameters of the model with Cross Validation and Bayesian optimization.
#'
#' @param rec The recipe object including local preprocessing.
#' @param model  The model object including the list of hyperparameters, engine and mode.
#' @param v Perform cross-validation by dividing the training data into v folds.
#' @param trainingData The training data.
#' @param gridNum Initial number of iterations to run before starting the optimization algorithm.
#' @param iter The maximum number of search iterations.
#' @param seed Seed for reproducible results.
#'
#' @importFrom magrittr %>%
#'
#' @export

bayesOptCV <- function(rec = NULL,
                       model = NULL,
                       v = NULL,
                       trainingData = NULL,
                       gridNum = NULL,
                       iter = NULL,
                       seed = NULL) {
  set.seed(seed = as.numeric(seed))
  tunedWorkflow <- workflows::workflow() %>%
    workflows::add_recipe(rec) %>%
    workflows::add_model(model)

  folds <- rsample::vfold_cv(trainingData, v = as.numeric(v), strata = rec$var_info$variable[rec$var_info$role == "outcome"])
  initial <- ifelse(model$engine == "kknn", gridNum, length(model$args) * gridNum)

  if (quo_name(model$args$mtry) == "tune()") {
    param <- tunedWorkflow %>%
      hardhat::extract_parameter_set_dials() %>%
      recipes::update(mtry = dials::finalize(mtry(), trainingData))

    print("ERROR with lightgbm pacakge: use 3.3.5 version not 4.2.0 version")
    print('devtools::install_version("lightgbm", version = "3.3.5", repos = "http://cran.us.r-project.org")')
    set.seed(seed = as.numeric(seed))
    result <-
      tunedWorkflow %>%
      tune::tune_bayes(folds, initial = initial, iter = iter, param_info = param)
  } else {
    set.seed(seed = as.numeric(seed))
    result <-
      tunedWorkflow %>%
      tune::tune_bayes(folds, initial = initial, iter = iter)
  }

  return(list(tunedWorkflow = tunedWorkflow, result = result))
}


#' fitting in best model
#'
#' @details
#' Get the bayesOptCV function's return value and fit the model.
#'
#' @param optResult  The result object of bayesOptCV
#' @param metric  Baseline metric for evaluating model performance (classification: "roc_auc" (default), "accuracy" / regression: "rmse" (default), "rsq")
#' @param model The model object including the list of hyperparameters, engine and mode.
#' @param formula formula for modeling
#' @param trainingData The training data.
#' @param splitedData The whole dataset including the information of each fold
#' @param modelName The name of model defined by the algorithm and engine selected by user
#'
#' @importFrom magrittr %>%
#' @importFrom dplyr mutate
#'
#' @export

fitBestModel <- function(optResult,
                         metric,
                         model,
                         formula,
                         trainingData,
                         splitedData,
                         modelName) {
  bestParams <- tune::select_best(optResult[[2]], metric)
  finalSpec <- tune::finalize_model(model, bestParams)

  finalModel <- finalSpec %>% fit(eval(parse(text = formula)), trainingData)

  finalFittedModel <-
    optResult[[1]] %>%
    workflows::update_model(finalSpec) %>%
    tune::last_fit(splitedData)

  finalFittedModel$.predictions[[1]] <- finalFittedModel$.predictions[[1]] %>%
    dplyr::mutate(model = modelName)

  return(list(finalModel = finalModel, finalFittedModel = finalFittedModel, bestParams = bestParams))
}
