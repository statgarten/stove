#' AUC-ROC Curve
#'
#' @details
#' ML 모델 리스트로부터 AUC-ROC Curve를 생성합니다.
#'
#' @param modelsList  ML 모델 리스트
#' @param targetVar  타겟 변수
#'
#' @import RColorBrewer cowplot ggplot2 yardstick grDevices
#' @importFrom dplyr group_by
#' @importFrom magrittr %>%
#' @name %>%
#' @rdname pipe
#'
#' @export

rocCurve <- function(modelsList, targetVar) {
  colors <- grDevices::colorRampPalette(c("#C70A80", "#FBCB0A", "#3EC70B", "#590696", "#37E2D5"))

  plot <- do.call(rbind, modelsList)[[5]] %>% ## rbind here does nothing
    do.call(rbind, .) %>%
    dplyr::group_by(model) %>%
    yardstick::roc_curve(
      truth = eval(parse(text = targetVar)),
      .pred_1,
      event_level = "second"
    ) %>%
    ggplot(
      aes(
        x = 1 - specificity,
        y = sensitivity,
        color = model
      )
    ) +
    labs(
      title = "ROC curve",
      x = "False Positive Rate (1-Specificity)",
      y = "True Positive Rate (Sensitivity)"
    ) +
    geom_line(size = 1.1) +
    geom_abline(slope = 1, intercept = 0, size = 0.5) +
    scale_color_manual(values = colors(length(modelsList))) +
    coord_fixed() +
    cowplot::theme_cowplot()

  return(plot)
}

#' Confusion matrix
#'
#' @details
#' ML 모델 리스트 내 특정 모델에 대해 Confusion matrix를 생성합니다.
#'
#' @param modelName  모델명
#' @param modelsList  ML 모델 리스트
#' @param targetVar  타겟 변수
#'
#' @import ggplot2 yardstick tune
#' @importFrom magrittr %>%
#' @name %>%
#' @rdname pipe
#'
#' @export

confusionMatrix <- function(modelName, modelsList, targetVar) {
  tmpDf <- modelsList[[modelName]] %>%
    tune::collect_predictions() %>%
    as.data.frame() %>%
    dplyr::select(targetVar, .pred_class)

  confDf <- stats::xtabs(~ tmpDf$.pred_class + tmpDf[[targetVar]])

  input.matrix <- data.matrix(confDf)
  confusion <- as.data.frame(as.table(input.matrix))
  colnames(confusion)[1] <- "y_pred"
  colnames(confusion)[2] <- "actual_y"
  colnames(confusion)[3] <- "Frequency"

  plot <- ggplot(confusion, aes(x = actual_y, y = y_pred, fill = Frequency)) +
    geom_tile() +
    geom_text(aes(label = Frequency)) +
    scale_x_discrete(name = "Actual Class") +
    scale_y_discrete(name = "Predicted Class") +
    geom_text(aes(label = Frequency), colour = "black") +
    scale_fill_continuous(high = "#E9BC09", low = "#F3E5AC")

  return(plot)
}

#' Regression plot
#'
#' @details
#' ML 모델 리스트 내 특정 모델에 대해 Regression plot를 생성합니다.
#'
#' @param modelName  모델명
#' @param modelsList  ML 모델 리스트
#' @param targetVar  타겟 변수
#'
#' @import yardstick tune ggplot2 ggrepel
#' @importFrom magrittr %>%
#' @name %>%
#' @rdname pipe
#'
#' @export

regressionPlot <- function(modelName, modelsList, targetVar) {
  tmpDf <- modelsList[[modelName]] %>%
    tune::collect_predictions()

  lims <- c(min(tmpDf[[targetVar]]), max(tmpDf[[targetVar]]))

  plot <- modelsList[[modelName]] %>%
    tune::collect_predictions() %>%
    ggplot(aes(x = eval(parse(text = targetVar)), y = modelsList[[modelName]]$.predictions[[1]][1]$.pred)) +
    theme(
      panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
      panel.background = element_blank(), axis.line = element_line(colour = "#C70A80")
    ) +
    labs(
      title = "Regression Plot (Actual vs Prediced)",
      x = "Actual Value",
      y = "Predicted Value"
    ) +
    geom_abline(color = "black", lty = 2) +
    geom_point(alpha = 0.8, colour = "#C70A80") +
    scale_x_continuous(limits = lims) +
    scale_y_continuous(limits = lims)

  return(plot)
}

#' Evaluation metrics for Classification
#'
#' @details
#'  ML 모델 리스트로부터 Classification 모델들에 대한 Evaluation metrics를 생성합니다.
#'
#' @param modelsList  ML 모델 리스트
#' @param targetVar  타겟 변수
#'
#' @import yardstick tune ggplot2
#' @importFrom data.table transpose
#' @importFrom magrittr %>%
#' @name %>%
#' @rdname pipe
#' @importFrom dplyr select mutate
#'
#' @export

evalMetricsC <- function(modelsList, targetVar) {
  table <- data.frame()
  custom_metrics <- yardstick::metric_set(
    yardstick::accuracy,
    yardstick::sens,
    yardstick::spec,
    yardstick::precision,
    yardstick::f_meas,
    yardstick::kap,
    yardstick::mcc
  )

  for (i in 1:length(modelsList)) {
    tmp <- custom_metrics(modelsList[[as.numeric(i)]] %>%
      tune::collect_predictions(),
    truth = eval(parse(text = targetVar)),
    estimate = .pred_class
    ) %>%
      dplyr::select(.estimate) %>%
      data.table::transpose() %>%
      dplyr::mutate(across(where(is.numeric), ~ round(., 3)))

    table <- rbind(table, tmp)
    rownames(table)[i] <- modelsList[[as.numeric(i)]][[5]][[1]]$model[1]
  }
  colnames(table) <- c("Accuracy", "Recall", "Specificity", "Precision", "F1-score", "Kappa", "MCC")

  return(table)
}

#' Evaluation metrics for Regression
#'
#' @details
#' ML 모델 리스트로부터 Regression 모델들에 대한 Evaluation metrics를 생성합니다.
#'
#' @param modelsList  ML 모델 리스트
#' @param targetVar  타겟 변수
#'
#' @import ggplot2
#' @importFrom magrittr %>%
#' @importFrom dplyr select mutate
#'
#' @export

evalMetricsR <- function(modelsList, targetVar) {
  table <- data.frame()

  custom_metrics <- yardstick::metric_set(
    yardstick::rmse,
    yardstick::rsq,
    yardstick::mae,
    yardstick::mase,
    yardstick::rpd
  )

  for (i in 1:length(modelsList)) {
    tmp <- custom_metrics(modelsList[[as.numeric(i)]] %>%
      tune::collect_predictions(),
    truth = eval(parse(text = targetVar)),
    estimate = .pred
    ) %>%
      dplyr::select(.estimate) %>%
      data.table::transpose() %>%
      dplyr::mutate(across(where(is.numeric), ~ round(., 3)))

    table <- rbind(table, tmp)
    rownames(table)[i] <- modelsList[[as.numeric(i)]][[5]][[1]]$model[1]
  }

  colnames(table) <- c("RMSE", "RSQ", "MAE", "MASE", "RPD")

  return(table)
}

#' clusteringVis
#'
#' @details
#' Deprecated
#'
#' @import cluster factoextra stats ggplot2
#'
#' @param data  data
#' @param model  model
#' @param nStart  nStart
#' @param maxK  maxK
#'
#' @export

clusteringVis <- function(data = NULL,
                          model = NULL,
                          maxK = "15",
                          nBoot = "100",
                          selectOptimal = "silhouette",
                          seedNum = "6471") {
  colors <- grDevices::colorRampPalette(c("#C70A80", "#FBCB0A", "#3EC70B", "#590696", "#37E2D5"))
  set.seed(as.numeric(seedNum))

  elbowPlot <- factoextra::fviz_nbclust(
    x = data,
    FUNcluster = stats::kmeans,
    method = "wss"
  )

  if (selectOptimal == "silhouette") {
    optimalK <- factoextra::fviz_nbclust(
      x = data,
      FUNcluster = stats::kmeans,
      method = selectOptimal,
      k.max = as.numeric(maxK),
      barfill = "slateblue",
      barcolor = "slateblue",
      linecolor = "slateblue"
    )
    cols <- colors(optimalK$data$clusters[which.max(optimalK$data$y)])
  } else if (selectOptimal == "gap_stat") {
    optimalK <- factoextra::fviz_nbclust(
      x = data,
      FUNcluster = stats::kmeans,
      method = selectOptimal,
      k.max = as.numeric(maxK),
      nboot = as.numeric(nBoot),
      barfill = "slateblue",
      barcolor = "slateblue",
      linecolor = "slateblue"
    )
    cols <- colors(optimalK$data$clusters[which.max(optimalK$data$gap)])
  }

  clustVis <- factoextra::fviz_cluster(
    object = model,
    data = data,
    palette = cols,
    geom = "point",
    ellipse.type = "convex",
    ggtheme = theme_bw()
  )

  return(list(elbowPlot = elbowPlot, silhouettePlot = optimalK, clustVis = clustVis))
}


#'
#' #' rmsePlot
#' #'
#' #' @details
#' #' rmsePlot
#' #'
#' #' @import ggplot2 cowplot
#' #' @import grDevices
#' #' @importFrom dplyr select mutate group_by
#' #' @importFrom tibble tibble
#' #'
#' #' @export
#'
#' rmsePlot <- function(modelsList = NULL,
#'                      targetVar = NULL
#'                      ) {
#'
#'   df <- do.call(rbind, modelsList)[[5]] %>% ## rbind here does nothing
#'     do.call(rbind, .) %>%
#'     dplyr::select(model, .pred, !!as.name(targetVar)) %>%
#'     dplyr::mutate(errorSq = (!!as.name(targetVar) - .pred)**2) %>%
#'     dplyr::group_by(model) %>%
#'     dplyr::mutate(rmse = sqrt(mean(errorSq)))
#'
#'   plotDf<- df[!duplicated(df$model),] %>%
#'     dplyr::select(model, rmse)
#'
#'   rmse_interval <- function(rmse, deg_free, p_lower = 0.025, p_upper = 0.975){
#'     tibble(.pred_lower = sqrt(deg_free / qchisq(p_upper, df = deg_free)) * rmse,
#'            .pred_upper = sqrt(deg_free / qchisq(p_lower, df = deg_free)) * rmse)
#'   }
#'
#'   plotDf$lower <- rmse_interval(plotDf$rmse, nrow(data_train))$.pred_lower
#'   plotDf$upper <- rmse_interval(plotDf$rmse, nrow(data_train))$.pred_upper
#'
#'   colors <- grDevices::colorRampPalette(c("#C70A80", "#FBCB0A", "#3EC70B", "#590696", "#37E2D5"))
#'
#'   rmsePlot <- ggplot(plotDf,aes(x = model)) +
#'     geom_errorbar(aes(ymin = lower, ymax = upper), size = 1) +
#'     geom_point(aes(y = rmse),size=3) +
#'     coord_flip() +
#'     theme_bw() +
#'     xlab('') +
#'     theme(legend.position='none') +
#'     scale_color_manual(values = colors(length(modelsList))) +
#'     cowplot::theme_cowplot()
#'
#'   return(rmsePlot)
#'
#' }



























