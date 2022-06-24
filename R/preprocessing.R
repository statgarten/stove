#' Train-Test Split
#'
#' @details
#' Train-Test Split
#'
#' @param data  data
#' @param strata  strata
#'
#' @import rsample
#'
#' @export

trainTestSplit <- function(data = data, target = NULL){
  data_split <- rsample::initial_split(data, strata = target)
  train <- rsample::training(data_split)
  test  <- rsample::testing(data_split)

  return(list(train, test, data_split))
}
