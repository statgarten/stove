#' Train-Test Split
#'
#' @details
#' Separate the entire data into a training set and a test set.
#'
#' @param data  Full data set with global preprocess completed.
#' @param target The target variable.
#' @param prop  Proportion of total data to be used as training data.
#' @param seed  Seed for reproducible results.
#'
#' @import rsample
#' @importFrom tidyselect all_of
#'
#' @export

trainTestSplit <- function(data = NULL,
                           target = NULL,
                           prop,
                           seed = "4814") {
  set.seed(seed = as.numeric(seed))
  dataSplit <- rsample::initial_split(data, strata = tidyselect::all_of(target), prop = as.numeric(prop))
  train <- rsample::training(dataSplit)
  test <- rsample::testing(dataSplit)

  return(list(train = train, test = test, dataSplit = dataSplit, target = target))
}


#' Preprocessing for cross validation
#'
#' @details
#' Define the local preprocessing method to be applied to the training data for each fold when the training data is divided into several folds.
#'
#' @param data  Training dataset to apply local preprocessing recipe.
#' @param formula formula for modeling
#' @param imputation If "imputation = TRUE", the model will be trained using cross-validation with imputation.
#' @param normalization If "normalization = TRUE", the model will be trained using cross-validation with normalization
#' @param nominalImputationType Imputation method for nominal variable (Option: mode(default), bag, knn)
#' @param numericImputationType Imputation method for numeric variable (Option: mean(default), bag, knn, linear, lower, median, roll)
#' @param normalizationType Normalization method (Option: range(default), center, normalization, scale)
#' @param seed seed
#'
#' @rawNamespace import(recipes, except = c(step))
#'
#' @export

prepForCV <- function(data = NULL,
                      formula = NULL,
                      imputation = FALSE,
                      normalization = FALSE,
                      nominalImputationType = "mode",
                      numericImputationType = "mean",
                      normalizationType = "range",
                      seed = "4814") {
  set.seed(seed = as.numeric(seed))

  # one-hot encoding
  result <- recipes::recipe(eval(parse(text = formula)), data = data) %>%
    recipes::step_dummy(recipes::all_nominal_predictors())

  # Imputation
  if (imputation == TRUE) {
    if (!is.null(nominalImputationType)) {
      cmd <- paste0("result <- result %>% recipes::step_impute_", nominalImputationType, "(recipes::all_nominal_predictors())")
      eval(parse(text = cmd))
    }
    if (!is.null(numericImputationType)) {
      cmd <- paste0("result <- result %>% recipes::step_impute_", numericImputationType, "(recipes::all_numeric_predictors())")
      eval(parse(text = cmd))
    }
  }

  # Normalization
  if (normalization == TRUE) {
    if (!is.null(normalizationType)) {
      cmd <- paste0("result <- result %>% recipes::step_", normalizationType, "(recipes::all_numeric_predictors())")
      eval(parse(text = cmd))
    }
  }

  return(result)
}
