#' Train-Test Split
#'
#' @details
#' Data를 Train set과 Test set으로 분리합니다.
#'
#' @param data  전처리가 완료된 전체 data
#' @param target 타겟 변수
#' @param prop  전체 데이터 중 훈련 데이터로 사용할 비율
#' @param seed  seed값 설정
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
#' Deprecated
#'
#' @param data  data
#' @param formula formula
#' @param imputationType imputationType
#' @param normalizationType normalizationType
#' @param imputation imputation
#' @param normalization normalization
#' @param seed seed
#'
#' @rawNamespace import(recipes, except = c(step))
#'
#' @export

## todo: make user to choose predictors

prepForCV <- function(data = NULL,
                      formula = NULL,
                      imputation = FALSE,
                      normalization = FALSE,
                      nominalImputationType = "mode", # mode(default), bag, knn
                      numericImputationType = "mean", # mean(default), bag, knn, linear, lower, median, roll
                      normalizationType = "range", # range(default), center, normalization, scale
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

  # # PCA
  # if (pca == TRUE) {
  #   result <- result %>%
  #     recipes::step_pca(all_numeric_predictors(),
  #                       threshold = eval(parse(text = pcaThres))) ## todo: make users to perform PCA for numeric var only or numeric except for binary
  # } else {
  #   # pass
  # }

  return(result)
}
