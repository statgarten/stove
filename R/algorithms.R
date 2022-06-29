#' Logistic Regression
#'
#' @details
#' 로지스틱 회귀 알고리즘 함수. 예측 변수들이 정규분포를 따르지 않아도 사용할 수 있습니다.
#' 그러나 이 알고리즘은 결과 변수가 선형적으로 구분되며, 예측 변수들의 값이 결과 변수와 선형 관계를
#' 가진다고 가정합니다. 만약 데이터가 이 가정을 충족하지 않는 경우 성능이 저하될 수 있습니다.
#' hyperparameters: penalty, mixture
#'
#' @param engine  engine
#' @param mode  mode
#'
#' @import parsnip
#'
#' @export

logisticRegression_phi <- function(engine = "glm",
                                   mode = "classification"){

  result <- parsnip::logistic_reg(penalty = tune(),
                                  mixture = tune()) %>%
    parsnip::set_engine(engine = engine) %>%
    parsnip::set_mode(mode = mode)

  return(result)
}

#' Linear Regression
#'
#' @details
#' Linear Regression
#' hyperparameters: penalty, mixture
#'
#' @param engine  engine
#' @param mode  mode
#'
#' @import parsnip
#'
#' @export

linearRegression_phi <- function(engine = "lm",
                                   mode = "regression"){

  result <- parsnip::linear_reg(penalty = tune(),
                                  mixture = tune()) %>%
    parsnip::set_engine(engine = engine) %>%
    parsnip::set_mode(mode = mode)

  return(result)
}


#' Random Forest
#'
#' @details
#' 랜덤 포레스트 알고리즘 함수. 여러개의 Decision Tree를 형성.
#' 새로운 데이터 포인트를 각 트리에 동시에 통과 시켜 각 트리가 분류한 결과에서 투표를 실시하여
#' 가장 많이 득표한 결과를 최종 분류 결과로 선택
#' hyperparameters: trees, min_n, mtry
#'
#' @param engine  engine
#' @param mode  mode
#'
#' @import parsnip
#'
#' @export

randomForest_phi <- function(engine = "randomForest",
                             mode = "classification"){

  result <- parsnip::rand_forest(trees = tune(), min_n = tune(), mtry = tune()) %>%
    parsnip::set_engine(engine = engine, importance = "impurity") %>%
    parsnip::set_mode(mode = mode)

  return(result)
}


#' K-Nearest Neighbors
#'
#' @details
#' KNN 알고리즘 함수.
#' 데이터로부터 거리가 가까운 K개의 다른 데이터의 레이블을 참조하여 분류하는 알고리즘
#' hyperparameters: neighbors
#'
#'
#' @param engine  engine
#' @param mode  mode
#'
#' @import parsnip
#' @import kknn
#'
#' @export

knn_phi <- function(engine = "kknn",
                    mode = "classification"){

  result <- parsnip::nearest_neighbor(neighbors = tune()) %>%
    parsnip::set_engine(engine = engine) %>%
    parsnip::set_mode(mode = mode)

  return(result)
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
#' @param engine  engine
#' @param mode  mode
#'
#' @import parsnip
#'
#' @export

mlp_phi <- function(engine = "nnet",
                    mode = "classification"){

  result <- parsnip::mlp(hidden_units = tune(),
                         penalty = tune(),
                         dropout = tune(),
                         epochs = tune(),
                         activation = tune(),
                         learn_rate = tune()) %>%
    parsnip::set_engine(engine = engine) %>%
    parsnip::set_mode(mode = mode)

  return(result)
}


#' Decision Tree
#'
#' @details
#' 의사결정나무 알고리즘 함수. 의사 결정 규칙 (Decision rule)을 나무 형태로 분류해 나가는 분석 기법을 말합니다.
#' hyperparameters: cost_complexity, tree_depth, min_n
#'
#' @param engine engine
#' @param mode mode
#'
#' @import parsnip
#'
#' @export

decisionTree_phi <- function(engine = "rpart",
                             mode = "classification"){

  result <- parsnip::decision_tree(cost_complexity = tune(),
                                   tree_depth = tune(),
                                   min_n = tune()) %>%
    parsnip::set_engine(engine = engine) %>%
    parsnip::set_mode(mode = mode)

  return(result)
}


#' Naive Bayes
#'
#' @details
#' Naive Bayes
#' hyperparameters: smoothness, Laplace
#'
#' @param engine engine
#' @param mode mode
#'
#' @import parsnip
#'
#' @export

naiveBayes_phi <- function(engine = "klaR",
                           mode = "classification"){

  result <- parsnip::naive_Bayes(smoothness = tune(),
                                 Laplace = tune()) %>%
    parsnip::set_engine(engine = engine) %>%
    parsnip::set_mode(mode = mode)

  return(result)
}


#' Light GBM
#'
#' @details
#' Light GBM
#' hyperparameters: mtry, min_n, tree_depth, loss_reduction, learn_rate, sample_size
#'
#' @param engine engine
#' @param mode mode
#'
#' @import parsnip
#'
#' @export

lightGbm_phi <- function(engine = "lightgbm",
                         mode = "classification"){

  result <- parsnip::boost_tree(
    mtry = tune(),
    trees = tune(),
    min_n = tune(),
    tree_depth = tune(),
    loss_reduction = tune(),
    learn_rate = tune(),
    sample_size = tune()
    ) %>%
    parsnip::set_engine(engine = engine) %>%
    parsnip::set_mode(mode = mode)

  return(result)
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
#'
#' @import stats
#' @import factoextra
#'
#' @export
#'

kMeansClustering_phi <- function(data,
                                 maxK = 10,
                                 nstart = 25,
                                 selectOptimal = "silhouette",
                                 seed_num = 6471){

  set.seed(seed_num)
  tmp_result <- factoextra::fviz_nbclust(data, stats::kmeans, method = selectOptimal, k.max = maxK)

  if(selectOptimal == "silhouette"){
    result_clust<-tmp_result$data
    optimalK <- as.numeric(result_clust$clusters[which.max(result_clust$y)])
  } else if (selectOptimal == "gap_stat"){
    result_clust<-tmp_result$data
    optimalK <- as.numeric(result_clust$clusters[which.max(result_clust$gap)])
  } else {
    stop("selectOptimal must be 'silhouette' or 'gap_stat'.")
  }

  result <- stats::kmeans(data, optimalK, nstart = nstart)

  return(result)
}
