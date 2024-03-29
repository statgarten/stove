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
  tmp <- do.call(rbind, modelsList)[[5]] %>%
    do.call(rbind, .) %>%
    dplyr::group_by(model)

  if (".pred_1" %in% colnames(tmp)) {
    colors <- grDevices::colorRampPalette(c("#C70A80", "#FBCB0A", "#3EC70B", "#590696", "#37E2D5"))

    plot <- do.call(rbind, modelsList)[[5]] %>%
      do.call(rbind, .) %>%
      dplyr::group_by(model) %>%
      yardstick::roc_curve(
        # truth = eval(parse(text = targetVar)), 아래로 변경
        truth = targetVar,
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
  } else {
    stop("`rocCurve()` supports only the results of the binary classification model.")
  }

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
    tmp <- custom_metrics(
      modelsList[[as.numeric(i)]] %>%
        tune::collect_predictions(),
      # truth = eval(parse(text = targetVar)), 아래로 변경
      truth = targetVar,
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
    tmp <- custom_metrics(
      modelsList[[as.numeric(i)]] %>%
        tune::collect_predictions(),
      truth = targetVar,
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



#' rmsePlot
#'
#' @details
#' rmsePlot
#'
#' @import ggplot2
#' @import grDevices
#' @importFrom dplyr select mutate group_by
#' @importFrom tibble tibble
#'
#' @export

plotRmseComparison <- function(tunedResultsList,
                               v = v,
                               iter = iter) {
  combined_rmse_df <- data.frame()
  model_name <- names(tunedResultsList)

  for (i in seq_along(tunedResultsList)) {
    iter_df_merge <- data.frame()
    for (j in seq(v + 1, v + v * iter, by = 1)) {
      # model's name
      custom_name <- model_name[i] %>%
        as.data.frame()
      colnames(custom_name) <- "model"

      # iteration
      iteration <- j - v %>%
        as.data.frame()
      colnames(iteration) <- "iteration"

      # rmse value
      rmse_value <- tunedResultsList[[i]]$result$.metrics[[j]] %>%
        dplyr::filter(.metric == "rmse") %>%
        dplyr::pull(.estimate) %>%
        as.data.frame()
      colnames(rmse_value) <- "rmse_value"

      tmp <- cbind(custom_name, iteration, rmse_value)
      iter_df_merge <- rbind(iter_df_merge, tmp)
    }
    combined_rmse_df <- rbind(combined_rmse_df, iter_df_merge)
  }

  rmse_summary <- combined_rmse_df %>%
    group_by(model) %>%
    dplyr::summarize(
      mean_rmse = mean(rmse_value),
      rmse_se = sd(rmse_value) / sqrt(n())
    ) %>%
    mutate(
      lower_bound = mean_rmse - 1.96 * rmse_se,
      upper_bound = mean_rmse + 1.96 * rmse_se
    )

  colors <- grDevices::colorRampPalette(c("#C70A80", "#FBCB0A", "#3EC70B", "#590696", "#37E2D5"))

  rmse_plot <- ggplot(rmse_summary, aes(x = model, y = mean_rmse, ymin = lower_bound, ymax = upper_bound, color = model)) +
    geom_point(size = 3) +
    geom_errorbar(width = 0.2) +
    scale_color_manual(values = colors(length(tunedResultsList))) +
    labs(
      title = "RMSE Comparison",
      x = "Model",
      y = "Mean RMSE"
    ) +
    cowplot::theme_cowplot() +
    theme(
      axis.title.x = element_blank(),
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      panel.grid.major.y = element_line(color = "grey", linetype = "solid"),
      panel.grid.minor.y = element_line(color = "grey", linetype = "dashed")
    )

  return(list(rmse_plot = rmse_plot, rmse_summary = rmse_summary, model_name = model_name))
}
