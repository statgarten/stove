#' Logistic Regression
#'
#' @details
#' 로지스틱 회귀 알고리즘 함수. 예측 변수들이 정규분포를 따르지 않아도 사용할 수 있습니다.
#' 그러나 이 알고리즘은 결과 변수가 선형적으로 구분되며, 예측 변수들의 값이 결과 변수와 선형 관계를
#' 갖는다고 가정합니다. 만약 데이터가 이 가정을 충족하지 않는 경우 성능이 저하될 수 있습니다.
#' hyperparameters: penalty, mixture
#'
#' @param engine  engine
#' @param mode  mode
#'
#' @importFrom magrittr %>%
#' @name %>%
#' @rdname pipe
#' @importFrom dials penalty mixture
#' @import parsnip
#' @import stats glmnet LiblineaR rstanarm
#'
#' @export

logisticRegression <- function(algo = "logistic Regression",
                               engine = "glm",
                               mode = "classification",
                               trainingData = NULL,
                               splitedData = NULL,
                               formula = NULL,
                               rec = NULL,
                               v = "5",
                               penaltyRangeMin = "0.001",
                               penaltyRangeMax = "1.0",
                               penaltyRangeLevels = "5",
                               mixtureRangeMin = "0.0",
                               mixtureRangeMax = "1.0",
                               mixtureRangeLevels = "5",
                               metric = "roc_auc") {
  penaltyRange <- c(as.numeric(penaltyRangeMin), as.numeric(penaltyRangeMax))
  mixtureRange <- c(as.numeric(mixtureRangeMin), as.numeric(mixtureRangeMax))

  if (engine == "glmnet" || engine == "liblinear" || engine == "brulee") {
    parameterGrid <- dials::grid_regular(
      dials::penalty(range = penaltyRange),
      dials::mixture(range = mixtureRange),
      levels = c(
        penalty = as.numeric(penaltyRangeLevels),
        mixture = as.numeric(mixtureRangeLevels)
      )
    )
    model <- parsnip::logistic_reg(
      penalty = tune(),
      mixture = tune()
    ) %>%
      parsnip::set_engine(engine = engine) %>%
      parsnip::set_mode(mode = mode) %>%
      parsnip::translate()

    gridSearchResult <- goophi::gridSearchCV(
      rec = rec,
      model = model,
      v = as.numeric(v),
      data = trainingData,
      parameterGrid = parameterGrid
    )

    finalized <- goophi::fitBestModel(
      gridSearchResult = gridSearchResult,
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

    gridSearchResult <- goophi::gridSearchCV(
      rec = rec,
      model = model,
      v = as.numeric(v),
      data = trainingData,
      parameterGrid = 10 # default value of param 'grid' in tune::tune_grid
    )

    finalized <- goophi::fitBestModel(
      gridSearchResult = gridSearchResult,
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
#' Linear Regression
#' hyperparameters: penalty, mixture
#'
#' @param algo algo
#' @param engine engine
#' @param mode mode
#' @param trainingData trainingData
#' @param splitedData splitedData
#' @param formula formula
#' @param rec rec
#' @param v v
#' @param penaltyRangeMin penaltyRangeMin
#' @param penaltyRangeMax penaltyRangeMax
#' @param penaltyRangeLevels penaltyRangeLevels
#' @param mixtureRangeMin mixtureRangeMin
#' @param mixtureRangeMax mixtureRangeMax
#' @param mixtureRangeLevels mixtureRangeLevels
#' @param metric metric
#'
#' @importFrom magrittr %>%
#' @name %>%
#' @rdname pipe
#' @importFrom dials penalty mixture
#' @import parsnip
#' @import stats glmnet rstanarm
#'
#' @export

linearRegression <- function(algo = "linear Regression",
                             engine = "lm",
                             mode = "regression",
                             trainingData = NULL,
                             splitedData = NULL,
                             formula = NULL,
                             rec = NULL,
                             v = "5",
                             penaltyRangeMin = "0.001",
                             penaltyRangeMax = "1.0",
                             penaltyRangeLevels = "5",
                             mixtureRangeMin = "0.0",
                             mixtureRangeMax = "1.0",
                             mixtureRangeLevels = "5",
                             metric = "rmse") {
  penaltyRange <- c(as.numeric(penaltyRangeMin), as.numeric(penaltyRangeMax))
  mixtureRange <- c(as.numeric(mixtureRangeMin), as.numeric(mixtureRangeMax))

  if (engine == "glmnet" || engine == "liblinear" || engine == "brulee") {
    parameterGrid <- dials::grid_regular(
      dials::penalty(range = penaltyRange),
      dials::mixture(range = mixtureRange),
      levels = c(
        penalty = as.numeric(penaltyRangeLevels),
        mixture = as.numeric(mixtureRangeLevels)
      )
    )

    model <- parsnip::linear_reg(
      penalty = tune(),
      mixture = tune()
    ) %>%
      parsnip::set_engine(engine = engine) %>%
      parsnip::set_mode(mode = mode) %>%
      parsnip::translate()

    gridSearchResult <- goophi::gridSearchCV(
      rec = rec,
      model = model,
      v = as.numeric(v),
      data = trainingData,
      parameterGrid = parameterGrid
    )

    finalized <- goophi::fitBestModel(
      gridSearchResult = gridSearchResult,
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

    gridSearchResult <- goophi::gridSearchCV(
      rec = rec,
      model = model,
      v = as.numeric(v),
      data = trainingData,
      parameterGrid = 10 # default value of param 'grid' in tune::tune_grid
    )

    finalized <- goophi::fitBestModel(
      gridSearchResult = gridSearchResult,
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
#' @param algo algo
#' @param engine engine
#' @param mode mode
#' @param trainingData trainingData
#' @param splitedData splitedData
#' @param formula formula
#' @param rec rec
#' @param v v
#' @param neighborsRangeMin neighborsRangeMin
#' @param neighborsRangeMax neighborsRangeMax
#' @param neighborsRangeLevels neighborsRangeLevels
#' @param metric metric
#'
#' @importFrom magrittr %>%
#' @name %>%
#' @rdname pipe
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
                v = "5",
                neighborsRangeMin = "1",
                neighborsRangeMax = "10",
                neighborsRangeLevels = "10",
                metric = "roc_auc") {
  neighborsRange <- c(as.numeric(neighborsRangeMin), as.numeric(neighborsRangeMax))

  parameterGrid <- dials::grid_regular(
    dials::neighbors(range = neighborsRange),
    levels = c(neighbors = as.numeric(neighborsRangeLevels))
  )

  model <- parsnip::nearest_neighbor(neighbors = tune()) %>%
    parsnip::set_engine(engine = engine) %>%
    parsnip::set_mode(mode = mode) %>%
    parsnip::translate()

  gridSearchResult <- goophi::gridSearchCV(
    rec = rec,
    model = model,
    v = as.numeric(v),
    data = trainingData,
    parameterGrid = parameterGrid
  )

  finalized <- goophi::fitBestModel(
    gridSearchResult = gridSearchResult,
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
#' @param algo algo
#' @param engine engine
#' @param mode mode
#' @param trainingData trainingData
#' @param splitedData splitedData
#' @param formula formula
#' @param rec rec
#' @param v v
#' @param smoothnessRangeMin smoothnessRangeMin
#' @param smoothnessRangeMax smoothnessRangeMax
#' @param smoothnessRangeLevels smoothnessRangeLevels
#' @param LaplaceRangeMin LaplaceRangeMin
#' @param LaplaceRangeMax LaplaceRangeMax
#' @param LaplaceRangeLevels LaplaceRangeLevels
#' @param metric metric
#'
#' @importFrom magrittr %>%
#' @name %>%
#' @rdname pipe
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
                       v = "5",
                       smoothnessRangeMin = "0.5",
                       smoothnessRangeMax = "1.5",
                       smoothnessRangeLevels = "3",
                       LaplaceRangeMin = "0.0",
                       LaplaceRangeMax = "3.0",
                       LaplaceRangeLevels = "4",
                       metric = "roc_auc") {
  smoothnessRange <- c(as.numeric(smoothnessRangeMin), as.numeric(smoothnessRangeMax))
  LaplaceRange <- c(as.numeric(LaplaceRangeMin), as.numeric(LaplaceRangeMax))

  parameterGrid <- dials::grid_regular(
    discrim::smoothness(range = smoothnessRange),
    dials::Laplace(range = LaplaceRange),
    levels = c(
      smoothness = as.numeric(smoothnessRangeLevels),
      Laplace = as.numeric(LaplaceRangeLevels)
    )
  )

  model <- parsnip::naive_Bayes(
    smoothness = tune(),
    Laplace = tune()
  ) %>%
    parsnip::set_engine(engine = engine) %>%
    parsnip::set_mode(mode = mode) %>%
    parsnip::translate()

  gridSearchResult <- goophi::gridSearchCV(
    rec = rec,
    model = model,
    v = as.numeric(v),
    data = trainingData,
    parameterGrid = parameterGrid
  )

  finalized <- goophi::fitBestModel(
    gridSearchResult = gridSearchResult,
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
#' @param algo algo
#' @param engine engine
#' @param mode mode
#' @param trainingData trainingData
#' @param splitedData splitedData
#' @param formula formula
#' @param rec rec
#' @param v v
#' @param treeDepthRangeMin treeDepthRangeMin
#' @param treeDepthRangeMax treeDepthRangeMax
#' @param treeDepthRangeLevels treeDepthRangeLevels
#' @param minNRangeMin minNRangeMin
#' @param minNRangeMax minNRangeMax
#' @param minNRangeLevels minNRangeLevels
#' @param costComplexityRangeMin costComplexityRangeMin
#' @param costComplexityRangeMax costComplexityRangeMax
#' @param costComplexityRangeLevels costComplexityRangeLevels
#'
#' @importFrom magrittr %>%
#' @name %>%
#' @rdname pipe
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
                         v = "5",
                         treeDepthRangeMin = "1",
                         treeDepthRangeMax = "15",
                         treeDepthRangeLevels = "3",
                         minNRangeMin = "2",
                         minNRangeMax = "40",
                         minNRangeLevels = "3",
                         costComplexityRangeMin = "-2.0",
                         costComplexityRangeMax = "-1.0",
                         costComplexityRangeLevels = "2",
                         metric = "roc_auc") {
  treeDepthRange <- c(as.numeric(treeDepthRangeMin), as.numeric(treeDepthRangeMax))
  minNRange <- c(as.numeric(minNRangeMin), as.numeric(minNRangeMax))
  costComplexityRange <- c(as.numeric(costComplexityRangeMin), as.numeric(costComplexityRangeMax))

  parameterGrid <- dials::grid_regular(
    dials::tree_depth(range = treeDepthRange),
    dials::min_n(range = minNRange),
    dials::cost_complexity(range = costComplexityRange),
    levels = c(
      tree_depth = as.numeric(treeDepthRangeLevels),
      min_n = as.numeric(minNRangeLevels),
      cost_complexity = as.numeric(costComplexityRangeLevels)
    )
  )

  model <- parsnip::decision_tree(
    cost_complexity = tune(),
    tree_depth = tune(),
    min_n = tune()
  ) %>%
    parsnip::set_engine(engine = engine) %>%
    parsnip::set_mode(mode = mode) %>%
    parsnip::translate()

  gridSearchResult <- goophi::gridSearchCV(
    rec = rec,
    model = model,
    v = as.numeric(v),
    data = trainingData,
    parameterGrid = parameterGrid
  )

  finalized <- goophi::fitBestModel(
    gridSearchResult = gridSearchResult,
    metric = metric,
    model = model,
    formula = formula,
    trainingData = trainingData,
    splitedData = splitedData,
    algo = paste0(algo, "_", engine)
  )

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
#' @param algo algo
#' @param engine engine
#' @param mode mode
#' @param trainingData trainingData
#' @param splitedData splitedData
#' @param formula formula
#' @param rec rec
#' @param v v
#' @param mtryRangeMin mtryRangeMin
#' @param mtryRangeMax mtryRangeMax
#' @param mtryRangeLevels mtryRangeLevels
#' @param treesRangeMin treesRangeMin
#' @param treesRangeMax treesRangeMax
#' @param treesRangeLevels treesRangeLevels
#' @param minNRangeMin minNRangeMin
#' @param minNRangeMax minNRangeMax
#' @param minNRangeLevels minNRangeLevels
#'
#' @importFrom magrittr %>%
#' @name %>%
#' @rdname pipe
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
                         v = "5",
                         mtryRangeMin = "1",
                         mtryRangeMax = "20",
                         mtryRangeLevels = "3",
                         treesRangeMin = "100",
                         treesRangeMax = "1000",
                         treesRangeLevels = "3",
                         minNRangeMin = "2",
                         minNRangeMax = "40",
                         minNRangeLevels = "3",
                         metric = "roc_auc") {
  mtryRange <- c(as.numeric(mtryRangeMin), as.numeric(mtryRangeMax))
  treesRange <- c(as.numeric(treesRangeMin), as.numeric(treesRangeMax))
  minNRange <- c(as.numeric(minNRangeMin), as.numeric(minNRangeMax))

  parameterGrid <- dials::grid_regular(
    dials::mtry(range = mtryRange),
    dials::trees(range = treesRange),
    dials::min_n(range = minNRange),
    levels = c(
      mtry = as.numeric(mtryRangeLevels),
      trees = as.numeric(treesRangeLevels),
      min_n = as.numeric(minNRangeLevels)
    )
  )

  model <- parsnip::rand_forest(
    trees = tune(),
    min_n = tune(),
    mtry = tune()
  ) %>%
    parsnip::set_engine(engine = engine) %>%
    parsnip::set_mode(mode = mode) %>%
    parsnip::translate()

  gridSearchResult <- goophi::gridSearchCV(
    rec = rec,
    model = model,
    v = as.numeric(v),
    data = trainingData,
    parameterGrid = parameterGrid
  )

  finalized <- goophi::fitBestModel(
    gridSearchResult = gridSearchResult,
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
#' @param engine engine
#' @param mode mode
#'
#' @importFrom magrittr %>%
#' @name %>%
#' @rdname pipe
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
                    v = "5",
                    treeDepthRangeMin = "1",
                    treeDepthRangeMax = "15",
                    treeDepthRangeLevels = "3",
                    treesRangeMin = "10",
                    treesRangeMax = "500",
                    treesRangeLevels = "5",
                    learnRateRangeMin = "-2.0",
                    learnRateRangeMax = "-1.0",
                    learnRateRangeLevels = "2",
                    mtryRangeMin = "1",
                    mtryRangeMax = "20",
                    mtryRangeLevels = "3",
                    minNRangeMin = "2",
                    minNRangeMax = "40",
                    minNRangeLevels = "3",
                    lossReductionRangeMin = "-1.0",
                    lossReductionRangeMax = "1.0",
                    lossReductionRangeLevels = "3",
                    sampleSizeRangeMin = "0.0",
                    sampleSizeRangeMax = "1.0",
                    sampleSizeRangeLevels = "3",
                    stopIter = "30",
                    metric = "roc_auc") {
  treeDepthRange <- c(as.numeric(treeDepthRangeMin), as.numeric(treeDepthRangeMax))
  treesRange <- c(as.numeric(treesRangeMin), as.numeric(treesRangeMax))
  learnRateRange <- c(as.numeric(learnRateRangeMin), as.numeric(learnRateRangeMax))
  mtryRange <- c(as.numeric(mtryRangeMin), as.numeric(mtryRangeMax))
  minNRange <- c(as.numeric(minNRangeMin), as.numeric(minNRangeMax))
  lossReductionRange <- c(as.numeric(lossReductionRangeMin), as.numeric(lossReductionRangeMax))
  sampleSizeRange <- c(as.numeric(sampleSizeRangeMin), as.numeric(sampleSizeRangeMax))
  stopIterRange <- c(as.numeric(stopIter), as.numeric(stopIter)) # constant

  parameterGrid <- dials::grid_regular(
    dials::tree_depth(range = treeDepthRange),
    dials::trees(range = treesRange),
    dials::learn_rate(range = learnRateRange),
    dials::mtry(range = mtryRange),
    dials::min_n(range = minNRange),
    dials::loss_reduction(range = lossReductionRange),
    dials::sample_size(range = sampleSizeRange),
    dials::stop_iter(range = stopIterRange),
    levels = c(
      tree_depth = as.numeric(treeDepthRangeLevels),
      trees = as.numeric(treesRangeLevels),
      learn_rate = as.numeric(learnRateRangeLevels),
      mtry = as.numeric(mtryRangeLevels),
      min_n = as.numeric(minNRangeLevels),
      loss_reduction = as.numeric(lossReductionRangeLevels),
      sample_size = as.numeric(sampleSizeRangeLevels),
      stop_iter = 1
    )
  )

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
    parsnip::set_engine(engine = engine, counts = FALSE) %>%
    parsnip::set_mode(mode = mode) %>%
    parsnip::translate()

  gridSearchResult <- goophi::gridSearchCV(
    rec = rec,
    model = model,
    v = as.numeric(v),
    data = trainingData,
    parameterGrid = parameterGrid
  )

  finalized <- goophi::fitBestModel(
    gridSearchResult = gridSearchResult,
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
#' @param engine engine
#' @param mode mode
#'
#' @importFrom magrittr %>%
#' @name %>%
#' @rdname pipe
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
                     v = "5",
                     treeDepthRangeMin = "5",
                     treeDepthRangeMax = "15",
                     treeDepthRangeLevels = "3",
                     treesRangeMin = "10",
                     treesRangeMax = "100",
                     treesRangeLevels = "2",
                     learnRateRangeMin = "-2.0",
                     learnRateRangeMax = "-1.0",
                     learnRateRangeLevels = "2",
                     mtryRangeMin = "1",
                     mtryRangeMax = "20",
                     mtryRangeLevels = "3",
                     minNRangeMin = "2",
                     minNRangeMax = "40",
                     minNRangeLevels = "3",
                     lossReductionRangeMin = "-1.0",
                     lossReductionRangeMax = "1.0",
                     lossReductionRangeLevels = "3",
                     metric = "roc_auc") {
  treeDepthRange <- c(as.numeric(treeDepthRangeMin), as.numeric(treeDepthRangeMax))
  treesRange <- c(as.numeric(treesRangeMin), as.numeric(treesRangeMax))
  learnRateRange <- c(as.numeric(learnRateRangeMin), as.numeric(learnRateRangeMax))
  mtryRange <- c(as.numeric(mtryRangeMin), as.numeric(mtryRangeMax))
  minNRange <- c(as.numeric(minNRangeMin), as.numeric(minNRangeMax))
  lossReductionRange <- c(as.numeric(lossReductionRangeMin), as.numeric(lossReductionRangeMax))

  parameterGrid <- dials::grid_regular(
    dials::tree_depth(range = treeDepthRange),
    dials::trees(range = treesRange),
    dials::learn_rate(range = learnRateRange),
    dials::mtry(range = mtryRange),
    dials::min_n(range = minNRange),
    dials::loss_reduction(range = lossReductionRange),
    levels = c(
      tree_depth = as.numeric(treeDepthRangeLevels),
      trees = as.numeric(treesRangeLevels),
      learn_rate = as.numeric(learnRateRangeLevels),
      mtry = as.numeric(mtryRangeLevels),
      min_n = as.numeric(minNRangeLevels),
      loss_reduction = as.numeric(lossReductionRangeLevels)
    )
  )

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

  gridSearchResult <- goophi::gridSearchCV(
    rec = rec,
    model = model,
    v = as.numeric(v),
    data = trainingData,
    parameterGrid = parameterGrid
  )

  finalized <- goophi::fitBestModel(
    gridSearchResult = gridSearchResult,
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
#' @param algo algo
#' @param engine engine
#' @param mode mode
#' @param trainingData trainingData
#' @param splitedData splitedData
#' @param formula formula
#' @param rec rec
#' @param v v
#' @param hiddenUnitsRangeMin hiddenUnitsRangeMin
#' @param hiddenUnitsRangeMax hiddenUnitsRangeMax
#' @param hiddenUnitsRangeLevels hiddenUnitsRangeLevels
#' @param penaltyRangeMin penaltyRangeMin
#' @param penaltyRangeMax penaltyRangeMax
#' @param penaltyRangeLevels penaltyRangeLevels
#' @param dropoutRangeMin dropoutRangeMin
#' @param dropoutRangeMax dropoutRangeMax
#' @param dropoutRangeLevels dropoutRangeLevels
#' @param epochsRangeMin epochsRangeMin
#' @param epochsRangeMax epochsRangeMax
#' @param epochsRangeLevels epochsRangeLevels
#' @param activation activation
#' @param learnRateRangeMin learnRateRangeMin
#' @param learnRateRangeMax learnRateRangeMax
#' @param learnRateRangeLevels learnRateRangeLevels
#'
#' @importFrom magrittr %>%
#' @name %>%
#' @rdname pipe
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
                v = "5",
                hiddenUnitsRangeMin = "1",
                hiddenUnitsRangeMax = "10",
                hiddenUnitsRangeLevels = "3",
                penaltyRangeMin = "0.001",
                penaltyRangeMax = "1.0",
                penaltyRangeLevels = "3",
                epochsRangeMin = "10",
                epochsRangeMax = "100",
                epochsRangeLevels = "2",
                # dropoutRangeMin = "0",
                # dropoutRangeMax = "1",
                # dropoutRangeLevels = "2",
                # activation = "linear", #"linear", "softmax", "relu", and "elu"
                # learnRateRangeMin = "0",
                # learnRateRangeMax = "1",
                # learnRateRangeLevels = "2",
                metric = "roc_auc") {
  hiddenUnitsRange <- c(as.numeric(hiddenUnitsRangeMin), as.numeric(hiddenUnitsRangeMax))
  penaltyRange <- c(as.numeric(penaltyRangeMin), as.numeric(penaltyRangeMax))
  epochsRange <- c(as.numeric(epochsRangeMin), as.numeric(epochsRangeMax))
  # dropoutRange <- c(as.numeric(dropoutRangeMin), as.numeric(dropoutRangeMax))
  # learnRateRange <- c(as.numeric(learnRateRangeMin), as.numeric(learnRateRangeMax))

  parameterGrid <- dials::grid_regular(
    dials::hidden_units(range = hiddenUnitsRange),
    dials::penalty(range = penaltyRange),
    dials::epochs(range = epochsRange),
    # dials::dropout(range = dropoutRange),
    # dials::learn_rate(range = learnRateRange),
    # dials::activation(values = activation),
    levels = c(
      hidden_units = as.numeric(hiddenUnitsRangeLevels),
      penalty = as.numeric(penaltyRangeLevels),
      epochs = as.numeric(epochsRangeLevels)
      # dropout = as.numeric(dropoutRangeLevels),
      # learn_rate = as.numeric(learnRateRangeLevels),
      # activation = 1
    )
  )

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

  gridSearchResult <- goophi::gridSearchCV(
    rec = rec,
    model = model,
    v = as.numeric(v),
    data = trainingData,
    parameterGrid = parameterGrid
  )

  finalized <- goophi::fitBestModel(
    gridSearchResult = gridSearchResult,
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
#' @param data data
#' @param maxK maxK
#' @param nstart nstart
#' @param selectOptimal selectOptimal
#' @param seed_num seed_num
#' @param iter.max iter.max
#' @param algorithm  algorithm
#'
#' @importFrom magrittr %>%
#' @name %>%
#' @rdname pipe
#' @import stats
#' @import factoextra
#'
#' @export
#'

kMeansClustering <- function(data,
                             maxK = "15",
                             nStart = "25",
                             iterMax = "10",
                             nBoot = "100",
                             algorithm = "Hartigan-Wong", ## "Hartigan-Wong", "Lloyd", "Forgy", "MacQueen"
                             selectOptimal = "silhouette", # silhouette, gap_stat
                             seedNum = "6471") {
  set.seed(as.numeric(seedNum))

  if (selectOptimal == "silhouette") {
    tmp_result <- factoextra::fviz_nbclust(
      x = data,
      FUNcluster = stats::kmeans,
      method = selectOptimal,
      k.max = as.numeric(maxK)
    )
    result_clust <- tmp_result$data

    optimalK <- as.numeric(result_clust$clusters[which.max(result_clust$y)])
  } else if (selectOptimal == "gap_stat") {
    tmp_result <- factoextra::fviz_nbclust(
      x = data,
      FUNcluster = stats::kmeans,
      method = selectOptimal,
      k.max = as.numeric(maxK),
      nboot = as.numeric(nBoot)
    )

    result_clust <- tmp_result$data

    optimalK <- as.numeric(result_clust$clusters[which.max(result_clust$gap)])
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

  return(result)
}
