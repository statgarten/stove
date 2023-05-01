#' Grid search with cross validation
#'
#' @details
#' 하이퍼파라미터를 탐색하는 Grid Search와 데이터 셋을 나누어 평가하는 cross validation을 함께 수행합니다.
#'
#' @param rec 데이터, 전처리 정보를 포함한 recipe object
#' @param model  hyperparameters, ngine, mode 정보가 포함된 model object
#' @param v v-fold cross validation을 진행 (default: 5, 각 fold 별로 30개 이상의 observations가 있어야 유효한 모델링 결과를 얻을 수 있습니다.)
#' @param trainingData 훈련데이터 셋
#' @param parameter_grid grid search를 수행할 때 각 hyperparameter의 값을 담은 object
#' @param seed seed값 설정
#'
#' @importFrom magrittr %>%
#'
#' @export

gridSearchCV <- function(rec = NULL,
                         model = NULL,
                         v = NULL,
                         trainingData = NULL,
                         parameterGrid = NULL,
                         seed = NULL) {
  tunedWorkflow <- workflows::workflow() %>%
    workflows::add_recipe(rec) %>%
    workflows::add_model(model)

  set.seed(seed = as.numeric(seed))
  result <- tune::tune_grid(tunedWorkflow,
    resamples = rsample::vfold_cv(trainingData, v = as.numeric(v)),
    grid = parameterGrid
  )

  return(list(tunedWorkflow = tunedWorkflow, result = result))
}

#' Bayesian optimization with cross validation
#'
#' @details
#' 교차검증 수행 과정에서, Bayesian optimization을 통해 모델의 하이퍼파라미터를 최적화합니다.
#'
#' @param rec 데이터, 전처리 정보를 포함한 recipe object
#' @param model  hyperparameters, ngine, mode 정보가 포함된 model object
#' @param v v-fold cross validation을 진행 (default: 5, 각 fold 별로 30개 이상의 observations가 있어야 유효한 모델링 결과를 얻을 수 있습니다.)
#' @param trainingData 훈련데이터 셋
#' @param initial 몇 개의 grid로
#' @param iter grid search를 수행할 때 각 hyperparameter의 값을 담은 object
#' @param seed seed값 설정
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

  folds <- rsample::vfold_cv(trainingData, v = as.numeric(v), strata = rec$var_info$variable[rec$var_info$role=="outcome"])
  initial <- ifelse(model$engine == "kknn", gridNum, length(model$args)*gridNum)

  if (quo_name(model$args$mtry) == "tune()") {
  param <- tunedWorkflow %>%
    hardhat::extract_parameter_set_dials() %>%
    recipes::update(mtry = dials::finalize(mtry(), trainingData))

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
#' gridSearchCV 함수 리턴값을 받아 가장 성능이 좋은 모델을 fitting합니다.
#'
#' @param optResult  gridSearchCV의 결과값
#' @param metric  모델의 성능을 평가할 기준지표 (classification: "roc_auc" (default), "accuracy" / regression: "rmse" (default), "rsq")
#' @param model hyperparameters, ngine, mode 정보가 포함된 model object
#' @param formula 모델링을 위한 수식
#' @param trainingData 훈련데이터 셋
#' @param splitedData train-test 데이터 분할 정보를 포함하고 있는 전체 데이터 셋
#' @param algo 사용자가 임의로 지정할 알고리즘명 (default: "linear Regression")
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
                         algo) {
  bestParams <- tune::select_best(optResult[[2]], metric)
  finalSpec <- tune::finalize_model(model, bestParams)

  finalModel <- finalSpec %>% fit(eval(parse(text = formula)), trainingData)

  finalFittedModel <-
    optResult[[1]] %>%
    workflows::update_model(finalSpec) %>%
    tune::last_fit(splitedData)

  finalFittedModel$.predictions[[1]] <- finalFittedModel$.predictions[[1]] %>%
    dplyr::mutate(model = algo)

  return(list(finalModel = finalModel, finalFittedModel = finalFittedModel, bestParams = bestParams))
}
