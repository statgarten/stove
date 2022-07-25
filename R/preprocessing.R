#' Train-Test Split
#'
#' @details
#' Train-Test Split
#'
#' @param data  data
#' @param target  target
#' @param prop  prop
#' @param seed  seed
#'
#' @import rsample
#' @importFrom tidyselect all_of
#'
#' @export

trainTestSplit <- function(data = data,
                           target = NULL,
                           prop,
                           seed = "4814"){
  set.seed(seed = as.numeric(seed))
  dataSplit <- rsample::initial_split(data, strata = tidyselect::all_of(target), prop = as.numeric(prop))
  train <- rsample::training(dataSplit)
  test  <- rsample::testing(dataSplit)

  return(list(train = train, test = test, dataSplit = dataSplit))
}


#' Preprocessing for cross validation
#'
#' @details
#' Preprocessing for cross validation
#'
#' @param data  data
#' @param formula  formula
#' @param imputationType imputationType
#' @param normalizationType normalizationType
#' @param pcaThres pcaThres
#' @param imputation imputation
#' @param normalization normalization
#' @param pca pca
#' @param seed seed
#'
#' @importFrom recipes recipe
#' @importFrom recipes step_impute_bag step_impute_knn step_impute_linear step_impute_lower step_impute_mean step_impute_median step_impute_mode step_impute_roll
#' @importFrom recipes step_center step_normalize step_range step_scale
#' @importFrom recipes step_pca
#'
#' @export

## todo: make user to choose predictors

prepForCV <- function(data,
                     formula,
                     imputationType = "mean",
                     normalizationType = "range", # min-max normalization as default
                     imputation = FALSE,
                     normalization = FALSE,
                     seed = "4814"){
  set.seed(seed = as.numeric(seed))
  result <- recipes::recipe(eval(parse(text = formula)), data = data) %>%
    recipes::step_dummy(recipes::all_nominal_predictors())

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

  # # PCA
  # if (pca == TRUE) {
  #   result <- result %>%
  #     recipes::step_pca(all_numeric(),
  #                       threshold = eval(parse(text = pcaThres))) ## todo: make users to perform PCA for numeric var only or numeric except for binary
  # } else {
  #   # pass
  # }

  return(result)
}
