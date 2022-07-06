#' Train-Test Split
#'
#' @details
#' Train-Test Split
#'
#' @param data  data
#' @param strata  strata
#' @param prop  prop
#'
#' @import rsample
#'
#' @export

trainTestSplit <- function(data = data, target = NULL, prop){
  data_split <- rsample::initial_split(data, strata = target, prop = as.numeric(prop))
  train <- rsample::training(data_split)
  test  <- rsample::testing(data_split)

  return(list(train, test, data_split))
}


#' Preprocessing
#'
#' @details
#' Preprocessing
#'
#' @param data  dataW
#' @param formula  formula
#' @param imputationType imputationType
#' @param normalizationType normalizationType
#' @param pcaThres pcaThres
#' @param imputation imputation
#' @param normalization normalization
#' @param pca pca
#'
#' @import recipes
#'
#' @export

## todo: make user to choose predictors

preprocessing <- function(data,
                     formula,
                     imputationType = "mean",
                     normalizationType = "range", # min-max normalization as default
                     pcaThres = "0.7", # string parameter for Shiny
                     imputation = TRUE,
                     normalization = TRUE,
                     pca = TRUE){

  result <- recipe(eval(parse(text = formula)), data = data)


  # Imputation
  if (imputation == TRUE) {
    if (imputationType == "bag") {
      result <- result %>% recipes::step_impute_bag(all_predictors())
    } else if (imputationType == "knn"){
      result <-result %>% recipes::step_impute_knn(all_predictors())
    } else if (imputationType == "linear"){
      result <- result %>% recipes::step_impute_linear(all_predictors())
    } else if (imputationType == "lower"){
      result <- result %>% recipes::step_impute_lower(all_predictors())
    } else if (imputationType == "mean"){
      result <- result %>% recipes::step_impute_mean(all_predictors())
    } else if (imputationType == "median"){
      result <- result %>% recipes::step_impute_median(all_predictors())
    } else if (imputationType == "mode"){
      result <- result %>% recipes::step_impute_mode(all_predictors())
    } else if (imputationType == "roll"){
      result <- result %>% recipes::step_impute_roll(all_predictors())
    } else {
      # pass
    }
  }

  # Normalization
  if (normalization == TRUE) {
    if (normalizationType == "center") {
      result <- result %>% recipes::step_center(all_numeric())
    } else if (normalizationType == "normalize") {
      result <- result %>% recipes::step_normalize(all_numeric())
    } else if (normalizationType == "range") {
      result <- result %>% recipes::step_range(all_numeric())
    } else if (normalizationType == "scale") {
      result <- result %>% recipes::step_scale(all_numeric())
    } else {
      # pass
    }
  }

  # PCA
  if (pca == TRUE) {
    result <- result %>%
      recipes::step_pca(all_numeric(),
                        threshold = eval(parse(text = pcaThres))) ## todo: make users to perform PCA for numeric var only or numeric except for binary
  } else {
    # pass
  }

  return(result)
}
