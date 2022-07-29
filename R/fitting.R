#' Grid Search with cross validation
#'
#' @details
#' Grid Search with cross validation // workflows rsample tune
#'
#' @param rec  rec
#' @param model  model
#' @param v v-fold CV
#' @param data data
#' @param parameter_grid parameter_grid
#' @param seed seed
#'
#' @importFrom magrittr %>%
#' @name %>%
#' @rdname pipe
#'
#' @export

gridSearchCV <- function(rec,
                         model,
                         v = "5", # 5-fold CV as default
                         data,
                         parameterGrid = 10,
                         seed = "4814") {
  set.seed(seed = as.numeric(seed))
  tunedWorkflow <- workflows::workflow() %>%
    workflows::add_recipe(rec) %>%
    workflows::add_model(model)

  result <- tune::tune_grid(tunedWorkflow,
    resamples = rsample::vfold_cv(data, v = as.numeric(v)),
    grid = parameterGrid
  ) # warnings

  return(list(tunedWorkflow = tunedWorkflow, result = result))
}


#' fitting in best model
#'
#' @details
#' fitting in best model // tune workflows
#'
#' @param gridSearchResult  gridSearchResult
#' @param metric  metric
#' @param model model
#' @param formula formula
#' @param trainingData trainingData
#' @param splitedData splitedData
#' @param algo algo
#'
#' @importFrom magrittr %>%
#' @name %>%
#' @rdname pipe
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
  bestParams <- tune::select_best(gridSearchResult[[2]], metric) ## metric 목록 print 되도록
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
