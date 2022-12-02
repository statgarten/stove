#' logistic Regression
#'
#' @details
#' 로지스틱 회귀 알고리즘 함수. 예측 변수들이 정규분포를 따르지 않아도 사용할 수 있습니다.
#' 그러나 이 알고리즘은 결과 변수가 선형적으로 구분되며, 예측 변수들의 값이 결과 변수와 선형 관계를
#' 갖는다고 가정합니다. 만약 데이터가 이 가정을 충족하지 않는 경우 성능이 저하될 수 있습니다.
#' 필요 hyperparameters: penalty, mixture
#'
#' @param algo 사용자가 임의로 지정할 알고리즘명 (default: "logistic Regression")
#' @param engine  모델을 생성할 때 사용할 패키지 ("glmnet" (default), "glm", "stan")
#' @param mode  분석 유형 ("classification" (default))
#' @param trainingData 훈련데이터 셋
#' @param splitedData train-test 데이터 분할 정보를 포함하고 있는 전체 데이터 셋
#' @param formula 모델링을 위한 수식
#' @param rec 데이터, 전처리 정보를 포함한 recipe object
#' @param v v-fold cross validation을 진행 (default: 5, 각 fold 별로 30개 이상의 observations가 있어야 유효한 모델링 결과를 얻을 수 있습니다.)
#' @param metric 모델의 성능을 평가할 기준지표 (classification: "roc_auc" (default), "accuracy" / regression: "rmse" (default), "rsq")
#' @param ... hyperparameters의 범위에 대한 Min, Max, Levels 값에 해당하는 파라미터를 지정합니다.
#'
#' @importFrom magrittr %>%
#' @importFrom dials penalty mixture
#' @import parsnip
#' @import stats glmnet LiblineaR rstanarm
#'
#' @export

logisticRegression <- function(algo = "logistic Regression",
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

  if (engine == "glmnet") {
    model <- parsnip::logistic_reg(
      penalty = tune(),
      mixture = tune()
    ) %>%
      parsnip::set_engine(engine = engine) %>%
      parsnip::set_mode(mode = mode) %>%
      parsnip::translate()

    bayesOptResult <- stove::bayesOptCV(
      rec = rec,
      model = model,
      v = as.numeric(v), # 5-fold CV as default
      trainingData = trainingData,
      gridNum = gridNum,
      iter = iter,
      seed = seed
    )

    finalized <- stove::fitBestModel(
      optResult = bayesOptResult,
      metric = metric,
      model = model,
      formula = formula,
      trainingData = trainingData,
      splitedData = splitedData,
      algo = paste0(algo, "_", engine)
    )
  } else {
    model <- parsnip::logistic_reg() %>%
      parsnip::set_engine(engine = engine) %>%
      parsnip::set_mode(mode = mode) %>%
      parsnip::translate()

    bayesOptResult <- stove::bayesOptCV(
      rec = rec,
      model = model,
      v = as.numeric(v), # 5-fold CV as default
      trainingData = trainingData,
      gridNum = gridNum,
      iter = iter,
      seed = seed
    )

    finalized <- stove::fitBestModel(
      optResult = bayesOptResult,
      metric = metric,
      model = model,
      formula = formula,
      trainingData = trainingData,
      splitedData = splitedData,
      algo = paste0(algo, "_", engine)
    )
  }

  return(finalized)
}


#' Linear Regression
#'
#' @details
#' 선형 회귀 알고리즘 함수. 선형회귀는 다음과 같은 가정을 합니다. 1) target - features 간의 선형성, 2) features 간 작은 다중공선성,
#' 3) 등분산성 가정, 4) 오차항의 정규분포, 5) 오차항 간 적은 상관성. 만약 데이터가 이 가정을 충족하지 않는 경우 성능이 저하될 수 있습니다.
#' hyperparameters: penalty, mixture
#'
#' @param algo 사용자가 임의로 지정할 알고리즘명 (default: "Linear Regression")
#' @param engine  모델을 생성할 때 사용할 패키지 ("glmnet" (default), "lm", "glm", "stan")
#' @param mode  분석 유형 ("regression" (default))
#' @param trainingData 훈련데이터 셋
#' @param splitedData train-test 데이터 분할 정보를 포함하고 있는 전체 데이터 셋
#' @param formula 모델링을 위한 수식
#' @param rec 데이터, 전처리 정보를 포함한 recipe object
#' @param v v-fold cross validation을 진행 (default: 5, 각 fold 별로 30개 이상의 observations가 있어야 유효한 모델링 결과를 얻을 수 있습니다.)
#' @param metric 모델의 성능을 평가할 기준지표 (classification: "roc_auc" (default), "accuracy" / regression: "rmse" (default), "rsq")
#' @param ... hyperparameters의 범위에 대한 Min, Max, Levels 값에 해당하는 파라미터를 지정합니다.
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

  if (engine == "glmnet" || engine == "liblinear" || engine == "brulee") {

    model <- parsnip::linear_reg(
      penalty = tune(),
      mixture = tune()
    ) %>%
      parsnip::set_engine(engine = engine) %>%
      parsnip::set_mode(mode = mode) %>%
      parsnip::translate()

    bayesOptResult <- stove::bayesOptCV(
      rec = rec,
      model = model,
      v = as.numeric(v), # 5-fold CV as default
      trainingData = trainingData,
      gridNum = gridNum,
      iter = iter,
      seed = seed
    )

    finalized <- stove::fitBestModel(
      optResult = bayesOptResult,
      metric = metric,
      model = model,
      formula = formula,
      trainingData = trainingData,
      splitedData = splitedData,
      algo = paste0(algo, "_", engine)
    )
  } else {
    model <- parsnip::linear_reg() %>%
      parsnip::set_engine(engine = engine) %>%
      parsnip::set_mode(mode = mode) %>%
      parsnip::translate()

    bayesOptResult <- stove::bayesOptCV(
      rec = rec,
      model = model,
      v = as.numeric(v), # 5-fold CV as default
      trainingData = trainingData,
      gridNum = gridNum,
      iter = iter,
      seed = seed
    )

    finalized <- stove::fitBestModel(
      optResult = bayesOptResult,
      metric = metric,
      model = model,
      formula = formula,
      trainingData = trainingData,
      splitedData = splitedData,
      algo = paste0(algo, "_", engine)
    )
  }

  return(finalized)
}


#' K-Nearest Neighbors
#'
#' @details
#' KNN 알고리즘 함수.
#' 데이터로부터 거리가 가까운 K개의 다른 데이터의 레이블을 참조하여 분류하는 알고리즘
#' hyperparameters: neighbors
#'
#' @param algo 사용자가 임의로 지정할 알고리즘명 (default: "KNN")
#' @param engine  모델을 생성할 때 사용할 패키지 ("kknn" (default))
#' @param mode  분석 유형 ("classification" (default), "regression")
#' @param trainingData 훈련데이터 셋
#' @param splitedData train-test 데이터 분할 정보를 포함하고 있는 전체 데이터 셋
#' @param formula 모델링을 위한 수식
#' @param rec 데이터, 전처리 정보를 포함한 recipe object
#' @param v v-fold cross validation을 진행 (default: 5, 각 fold 별로 30개 이상의 observations가 있어야 유효한 모델링 결과를 얻을 수 있습니다.)
#' @param metric 모델의 성능을 평가할 기준지표 (classification: "roc_auc" (default), "accuracy" / regression: "rmse" (default), "rsq")
#' @param ... hyperparameters의 범위에 대한 Min, Max, Levels 값에 해당하는 파라미터를 지정합니다.
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

  bayesOptResult <- stove::bayesOptCV(
    rec = rec,
    model = model,
    v = as.numeric(v), # 5-fold CV as default
    trainingData = trainingData,
    gridNum = gridNum,
    iter = iter,
    seed = seed
  )

  finalized <- stove::fitBestModel(
    optResult = bayesOptResult,
    metric = metric,
    model = model,
    formula = formula,
    trainingData = trainingData,
    splitedData = splitedData,
    algo = paste0(algo, "_", engine)
  )

  return(finalized)
}

#' Naive Bayes
#'
#' @details
#' Naive Bayes
#' hyperparameters: smoothness, Laplace
#'
#' @param algo 사용자가 임의로 지정할 알고리즘명 (default: "Naive Bayes")
#' @param engine  모델을 생성할 때 사용할 패키지 ("klaR" (default), naivebayes)
#' @param mode  분석 유형 ("classification" (default))
#' @param trainingData 훈련데이터 셋
#' @param splitedData train-test 데이터 분할 정보를 포함하고 있는 전체 데이터 셋
#' @param formula 모델링을 위한 수식
#' @param rec 데이터, 전처리 정보를 포함한 recipe object
#' @param v v-fold cross validation을 진행 (default: 5, 각 fold 별로 30개 이상의 observations가 있어야 유효한 모델링 결과를 얻을 수 있습니다.)
#' @param metric 모델의 성능을 평가할 기준지표 (classification: "roc_auc" (default), "accuracy" / regression: "rmse" (default), "rsq")
#' @param ... hyperparameters의 범위에 대한 Min, Max, Levels 값에 해당하는 파라미터를 지정합니다.
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

  bayesOptResult <- stove::bayesOptCV(
    rec = rec,
    model = model,
    v = as.numeric(v), # 5-fold CV as default
    trainingData = trainingData,
    gridNum = gridNum,
    iter = iter,
    seed = seed
  )

  finalized <- stove::fitBestModel(
    optResult = bayesOptResult,
    metric = metric,
    model = model,
    formula = formula,
    trainingData = trainingData,
    splitedData = splitedData,
    algo = paste0(algo, "_", engine)
  )

  return(finalized)
}

#' Decision Tree
#'
#' @details
#' 의사결정나무 알고리즘 함수. 의사 결정 규칙 (Decision rule)을 나무 형태로 분류해 나가는 분석 기법을 말합니다.
#' hyperparameters:
#' tree_depth: 최종 예측값에 다다르기까지 몇 번 트리를 분할할지 설정합니다.
#' min_n: 트리를 분할하기 위해 필요한 관측값의 최소 개수를 설정합니다.
#' cost_complexity: 트리 분할을 위해 필요한 비용을 설정합니다. 0일 경우, 가능한 모든 분할이 수행됩니다.
#'
#' @param algo 사용자가 임의로 지정할 알고리즘명 (default: "Decision Tree")
#' @param engine  모델을 생성할 때 사용할 패키지 ("rpart" (default), "C50", "partykit")
#' @param mode  분석 유형 ("classification" (default), "regression")
#' @param trainingData 훈련데이터 셋
#' @param splitedData train-test 데이터 분할 정보를 포함하고 있는 전체 데이터 셋
#' @param formula 모델링을 위한 수식
#' @param rec 데이터, 전처리 정보를 포함한 recipe object
#' @param v v-fold cross validation을 진행 (default: 5, 각 fold 별로 30개 이상의 observations가 있어야 유효한 모델링 결과를 얻을 수 있습니다.)
#' @param metric 모델의 성능을 평가할 기준지표 (classification: "roc_auc" (default), "accuracy" / regression: "rmse" (default), "rsq")
#' @param ... hyperparameters의 범위에 대한 Min, Max, Levels 값에 해당하는 파라미터를 지정합니다.
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

  if (engine == "rpart"){

    model <- parsnip::decision_tree(
      cost_complexity = tune(),
      tree_depth = tune(),
      min_n = tune()
    ) %>%
      parsnip::set_engine(engine = engine) %>%
      parsnip::set_mode(mode = mode) %>%
      parsnip::translate()

    bayesOptResult <- stove::bayesOptCV(
      rec = rec,
      model = model,
      v = as.numeric(v), # 5-fold CV as default
      trainingData = trainingData,
      gridNum = gridNum,
      iter = iter,
      seed = seed
    )

    finalized <- stove::fitBestModel(
      optResult = bayesOptResult,
      metric = metric,
      model = model,
      formula = formula,
      trainingData = trainingData,
      splitedData = splitedData,
      algo = paste0(algo, "_", engine)
    )
  } else if (engine == "C5.0") {
    minNRange <- c(as.numeric(minNRangeMin), as.numeric(minNRangeMax))

    parameterGrid <- dials::grid_regular(
      dials::min_n(range = minNRange),
      levels = c(
        min_n = as.numeric(minNRangeLevels)
      )
    )

    model <- parsnip::decision_tree(
      min_n = tune()
    ) %>%
      parsnip::set_engine(engine = engine) %>%
      parsnip::set_mode(mode = mode) %>%
      parsnip::translate()

    bayesOptResult <- stove::bayesOptCV(
      rec = rec,
      model = model,
      v = as.numeric(v), # 5-fold CV as default
      trainingData = trainingData,
      gridNum = gridNum,
      iter = iter,
      seed = seed
    )

    finalized <- stove::fitBestModel(
      optResult = bayesOptResult,
      metric = metric,
      model = model,
      formula = formula,
      trainingData = trainingData,
      splitedData = splitedData,
      algo = paste0(algo, "_", engine)
    )
  } else { # partykit

    model <- parsnip::decision_tree(
      tree_depth = tune(),
      min_n = tune()
    ) %>%
      parsnip::set_engine(engine = engine) %>%
      parsnip::set_mode(mode = mode) %>%
      parsnip::translate()

    bayesOptResult <- stove::bayesOptCV(
      rec = rec,
      model = model,
      v = as.numeric(v), # 5-fold CV as default
      trainingData = trainingData,
      gridNum = gridNum,
      iter = iter,
      seed = seed
    )

    finalized <- stove::fitBestModel(
      optResult = bayesOptResult,
      metric = metric,
      model = model,
      formula = formula,
      trainingData = trainingData,
      splitedData = splitedData,
      algo = paste0(algo, "_", engine)
    )
  }

  return(finalized)
}


#' Random Forest
#'
#' @details
#' 랜덤 포레스트 알고리즘 함수. 여러개의 Decision Tree를 형성.
#' 새로운 데이터 포인트를 각 트리에 동시에 통과 시켜 각 트리가 분류한 결과에서 투표를 실시하여
#' 가장 많이 득표한 결과를 최종 분류 결과로 선택
#' hyperparameters:
#' trees: 결정트리의 개수를 지정합니다.
#' min_n: 트리를 분할하기 위해 필요한 관측값의 최소 개수를 설정합니다.
#' mtry: 트리를 분할하기 위해 필요한 feature의 수를 설정합니다.
#'
#' @param algo 사용자가 임의로 지정할 알고리즘명 (default: "Random Forest")
#' @param engine  모델을 생성할 때 사용할 패키지 ("rpart" (default), "randomForest", "partykit")
#' @param mode  분석 유형 ("classification" (default), "regression")
#' @param trainingData 훈련데이터 셋
#' @param splitedData train-test 데이터 분할 정보를 포함하고 있는 전체 데이터 셋
#' @param formula 모델링을 위한 수식
#' @param rec 데이터, 전처리 정보를 포함한 recipe object
#' @param v v-fold cross validation을 진행 (default: 5, 각 fold 별로 30개 이상의 observations가 있어야 유효한 모델링 결과를 얻을 수 있습니다.)
#' @param metric 모델의 성능을 평가할 기준지표 (classification: "roc_auc" (default), "accuracy" / regression: "rmse" (default), "rsq")
#' @param ... hyperparameters의 범위에 대한 Min, Max, Levels 값에 해당하는 파라미터를 지정합니다.
#'
#' @importFrom magrittr %>%


#' @importFrom dials mtry trees min_n
#' @import parsnip
#' @import ranger partykit
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

  bayesOptResult <- stove::bayesOptCV(
    rec = rec,
    model = model,
    v = as.numeric(v), # 5-fold CV as default
    trainingData = trainingData,
    gridNum = gridNum,
    iter = iter,
    seed = seed
  )

  finalized <- stove::fitBestModel(
    optResult = bayesOptResult,
    metric = metric,
    model = model,
    formula = formula,
    trainingData = trainingData,
    splitedData = splitedData,
    algo = paste0(algo, "_", engine)
  )

  return(finalized)
}


#' XGBoost
#'
#' @details
#' XGBoost
#'
#' @param algo 사용자가 임의로 지정할 알고리즘명 (default: "XGBoost")
#' @param engine  모델을 생성할 때 사용할 패키지 ("xgboost" (default))
#' @param mode  분석 유형 ("classification" (default), "regression")
#' @param trainingData 훈련데이터 셋
#' @param splitedData train-test 데이터 분할 정보를 포함하고 있는 전체 데이터 셋
#' @param formula 모델링을 위한 수식
#' @param rec 데이터, 전처리 정보를 포함한 recipe object
#' @param v v-fold cross validation을 진행 (default: 5, 각 fold 별로 30개 이상의 observations가 있어야 유효한 모델링 결과를 얻을 수 있습니다.)
#' @param metric 모델의 성능을 평가할 기준지표 (classification: "roc_auc" (default), "accuracy" / regression: "rmse" (default), "rsq")
#' @param ... hyperparameters의 범위에 대한 Min, Max, Levels 값에 해당하는 파라미터를 지정합니다.
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
    sample_size = tune(),
    stop_iter = tune()
  ) %>%
    parsnip::set_engine(engine = engine) %>%
    parsnip::set_mode(mode = mode) %>%
    parsnip::translate()

  bayesOptResult <- stove::bayesOptCV(
    rec = rec,
    model = model,
    v = as.numeric(v), # 5-fold CV as default
    trainingData = trainingData,
    gridNum = gridNum,
    iter = iter,
    seed = seed
  )

  finalized <- stove::fitBestModel(
    optResult = bayesOptResult,
    metric = metric,
    model = model,
    formula = formula,
    trainingData = trainingData,
    splitedData = splitedData,
    algo = paste0(algo, "_", engine)
  )

  return(finalized)
}

#' Light GBM
#'
#' @details
#' Light GBM
#'
#' @param algo 사용자가 임의로 지정할 알고리즘명 (default: "lightGBM")
#' @param engine  모델을 생성할 때 사용할 패키지 ("lightgbm" (default))
#' @param mode  분석 유형 ("classification" (default), "regression")
#' @param trainingData 훈련데이터 셋
#' @param splitedData train-test 데이터 분할 정보를 포함하고 있는 전체 데이터 셋
#' @param formula 모델링을 위한 수식
#' @param rec 데이터, 전처리 정보를 포함한 recipe object
#' @param v v-fold cross validation을 진행 (default: 5, 각 fold 별로 30개 이상의 observations가 있어야 유효한 모델링 결과를 얻을 수 있습니다.)
#' @param metric 모델의 성능을 평가할 기준지표 (classification: "roc_auc" (default), "accuracy" / regression: "rmse" (default), "rsq")
#' @param ... hyperparameters의 범위에 대한 Min, Max, Levels 값에 해당하는 파라미터를 지정합니다.
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
    parsnip::set_engine(engine = engine) %>%
    parsnip::set_mode(mode = mode) %>%
    parsnip::translate()

  bayesOptResult <- stove::bayesOptCV(
    rec = rec,
    model = model,
    v = as.numeric(v), # 5-fold CV as default
    trainingData = trainingData,
    gridNum = gridNum,
    iter = iter,
    seed = seed
  )

  finalized <- stove::fitBestModel(
    optResult = bayesOptResult,
    metric = metric,
    model = model,
    formula = formula,
    trainingData = trainingData,
    splitedData = splitedData,
    algo = paste0(algo, "_", engine)
  )

  return(finalized)
}

#' neural network
#'
#' @details
#' neural network 알고리즘 함수.
#' neural network 모델은 생물학적인 뉴런을 수학적으로 모델링한 것.
#' 여러개의 뉴런으로부터 입력값을 받아서 세포체에 저장하다가 자신의 용량을 넘어서면 외부로 출력값을 내보내는 것처럼,
#' 인공신경망 뉴런은 여러 입력값을 받아서 일정 수준이 넘어서면 활성화되어 출력값을 내보낸다.
#' hyperparameters: hidden_units, penalty, dropout, epochs, activation, learn_rate
#'
#' @param algo 사용자가 임의로 지정할 알고리즘명 (default: "MLP")
#' @param engine  모델을 생성할 때 사용할 패키지 ("nnet" (default))
#' @param mode  분석 유형 ("classification" (default), "regression")
#' @param trainingData 훈련데이터 셋
#' @param splitedData train-test 데이터 분할 정보를 포함하고 있는 전체 데이터 셋
#' @param formula 모델링을 위한 수식
#' @param rec 데이터, 전처리 정보를 포함한 recipe object
#' @param v v-fold cross validation을 진행 (default: 5, 각 fold 별로 30개 이상의 observations가 있어야 유효한 모델링 결과를 얻을 수 있습니다.)
#' @param metric 모델의 성능을 평가할 기준지표 (classification: "roc_auc" (default), "accuracy" / regression: "rmse" (default), "rsq")
#' @param ... hyperparameters의 범위에 대한 Min, Max, Levels 값에 해당하는 파라미터를 지정합니다.
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
    epochs = tune(),
    # dropout = tune(),
    # activation = tune(),
    # learn_rate = tune()
  ) %>%
    parsnip::set_engine(engine = engine) %>%
    parsnip::set_mode(mode = mode) %>%
    parsnip::translate()

  bayesOptResult <- stove::bayesOptCV(
    rec = rec,
    model = model,
    v = as.numeric(v), # 5-fold CV as default
    trainingData = trainingData,
    gridNum = gridNum,
    iter = iter,
    seed = seed
  )

  finalized <- stove::fitBestModel(
    optResult = bayesOptResult,
    metric = metric,
    model = model,
    formula = formula,
    trainingData = trainingData,
    splitedData = splitedData,
    algo = paste0(algo, "_", engine)
  )

  return(finalized)
}

#' K means clustering
#'
#' @details
#' K means clustering
#' selectOptimal: silhouette, gap_stat
#' hyperparameters: maxK, nstart
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
