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
