#' Grid Search with cross validation
#'
#' @details
#' Grid Search with cross validation // workflows rsample tune
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
                         v = "5", # 5-fold CV as default
                         trainingData = NULL,
                         parameterGrid = 10,
                         seed = "4814") {
  set.seed(seed = as.numeric(seed))
  tunedWorkflow <- workflows::workflow() %>%
    workflows::add_recipe(rec) %>%
    workflows::add_model(model)

  result <- tune::tune_grid(tunedWorkflow,
    resamples = rsample::vfold_cv(trainingData, v = as.numeric(v)),
    grid = parameterGrid
  ) # warnings

  return(list(tunedWorkflow = tunedWorkflow, result = result))
}


#' fitting in best model
#'
#' @details
#' fitting in best model // tune workflows
#'
#' @param gridSearchResult  gridSearchCV의 결과값
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

fitBestModel <- function(gridSearchResult,
                         metric,
                         model,
                         formula,
                         trainingData,
                         splitedData,
                         algo) {
  bestParams <- tune::select_best(gridSearchResult[[2]], metric)
  finalSpec <- tune::finalize_model(model, bestParams)

  finalModel <- finalSpec %>% fit(eval(parse(text = formula)), trainingData)

  finalFittedModel <-
    gridSearchResult[[1]] %>%
    workflows::update_model(finalSpec) %>%
    tune::last_fit(splitedData)

  finalFittedModel$.predictions[[1]] <- finalFittedModel$.predictions[[1]] %>%
    dplyr::mutate(model = algo)

  return(list(finalModel = finalModel, finalFittedModel = finalFittedModel))
}
