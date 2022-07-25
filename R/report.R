#' AUC-ROC Curve
#'
#' @details
#' AUC-ROC Curve
#'
#' @param models_list  models_list
#' @param targetVar  targetVar
#'
#' @import RColorBrewer
#' @import cowplot
#' @import ggplot2
#' @import yardstick
#' @importFrom dplyr group_by
#' @importFrom magrittr %>%
#' @name %>%
#' @rdname pipe
#'
#' @export

rocCurve <- function(modelsList, targetVar){
  colors = RColorBrewer::brewer.pal(7, "Set2")[1:8] # maximum 8 models?

  plot <- do.call(rbind, modelsList)[[5]] %>% ## rbind here does nothing
    do.call(rbind, .) %>%
    dplyr::group_by(model) %>%
    yardstick::roc_curve(truth = eval(parse(text = targetVar)),
                         .pred_1,
                         event_level = 'second') %>%
    ggplot2::ggplot(
      aes(
        x = 1-specificity,
        y = sensitivity,
        color = model
      )
    ) +
    ggplot2::labs(title = "ROC curve",
         x = "False Positive Rate (1-Specificity)",
         y = "True Positive Rate (Sensitivity)") +
    ggplot2::geom_line(size = 1.1) +
    ggplot2::geom_abline(slope = 1, intercept = 0, size = 0.5) +
    ggplot2::scale_color_manual(values = colors) +
    ggplot2::coord_fixed() +
    cowplot::theme_cowplot()

  return(plot)
}


#' Confusion matrix
#'
#' @details
#' Confusion matrix
#'
#' @param modelName  modelName
#' @param modelsList  modelsList
#' @param targetVar  targetVar
#'
#' @importFrom magrittr %>%
#' @name %>%
#' @rdname pipe
#' @import yardstick
#' @import tune
#' @import ggplot2
#'
#' @export

confusionMatrix <- function(modelName, modelsList, targetVar){

  tmpDf<- models_list[[modelName]] %>%
    tune::collect_predictions() %>%
    as.data.frame() %>%
    dplyr::select(targetVar, .pred_class)

  confDf <- xtabs(~tmpDf$.pred_class + tmpDf[[targetVar]])

  input.matrix <- data.matrix(confDf)
  confusion <- as.data.frame(as.table(input.matrix))
  colnames(confusion)[1] <- "y_pred"
  colnames(confusion)[2] <- "actual_y"
  colnames(confusion)[3] <- "Frequency"

  plot <- ggplot(confusion, aes(x = actual_y, y = y_pred, fill = Frequency)) + geom_tile() +
    geom_text(aes(label=Frequency)) +
    scale_x_discrete(name="Actual Class") + scale_y_discrete(name="Predicted Class") +
    geom_text(aes(label = Frequency),colour = "white") +
    scale_fill_continuous(high = "#132B43", low = "#56B1F7")

  return(plot)

}

#' Regression plot
#'
#' @details
#' Regression plot
#'
#' @param modelName  modelName
#' @param modelsList  modelsList
#' @param targetVar  targetVar
#'
#' @importFrom magrittr %>%
#' @name %>%
#' @rdname pipe
#' @import yardstick
#' @import tune
#' @import ggplot2
#' @import ggrepel
#'
#' @export

regressionPlot <- function(modelName, modelsList, targetVar){

  tmpDf <-  models_list[[modelName]] %>%
    tune::collect_predictions()

  lims <- c(min(tmpDf[[targetVar]]),max(tmpDf[[targetVar]]))

  rgplot <- models_list[[modelName]] %>%
    tune::collect_predictions() %>%
    ggplot2::ggplot(aes(x = eval(parse(text = targetVar)), y = models_list[[modelName]]$.predictions[[1]][1]$.pred)) +
    ggplot2::labs(title = "Regression Plot (Truth vs Prediced)",
                  x = "Truth",
                  y = "Predicted") +
    ggplot2::geom_abline(color = "gray50", lty = 2) +
    ggplot2::geom_point(alpha = 0.5) +
    scale_x_continuous(limits = lims) +
    scale_y_continuous(limits = lims)

  return(rgplot)
}


#' Evaluation metrics for Classification
#'
#' @details
#' Evaluation metrics for Classification
#'
#' @param modelsList  modelsList
#' @param targetVar  targetVar
#'
#' @importFrom magrittr %>%
#' @name %>%
#' @rdname pipe
#' @import yardstick
#' @import tune
#' @import ggplot2
#' @import data.table
#' @importFrom dplyr select mutate
#'
#' @export

evalMetricsC <- function(modelsList, targetVar){

  table <- data.frame()
  custom_metrics <- yardstick::metric_set(yardstick::accuracy,
                                          yardstick::sens,
                                          yardstick::spec,
                                          yardstick::precision,
                                          yardstick::f_meas,
                                          yardstick::kap,
                                          yardstick::mcc
  )

  for (i in 1:length(modelsList)) {
    tmp <- custom_metrics(models_list[[as.numeric(i)]] %>%
                          tune::collect_predictions(),
                        truth = eval(parse(text = targetVar)),
                        estimate = .pred_class) %>%
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
#' Evaluation metrics for Regression
#'
#' @param modelsList  modelsList
#' @param targetVar  targetVar
#'
#' @importFrom magrittr %>%
#' @name %>%
#' @rdname pipe
#' @import yardstick
#' @import tune
#' @import ggplot2
#' @import data.table
#' @importFrom dplyr select mutate
#'
#' @export

evalMetricsR <- function(modelsList, targetVar){

  table <- data.frame()

  custom_metrics <- yardstick::metric_set(yardstick::rmse,
                                          yardstick::rsq,
                                          yardstick::mae,
                                          yardstick::mase,
                                          yardstick::rpd
  )

  for (i in 1:length(models_list)) {
    tmp <- custom_metrics(modelsList[[as.numeric(i)]] %>%
                            tune::collect_predictions(),
                          truth = eval(parse(text = targetVar)),
                          estimate = .pred) %>%
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
#' clusteringVis
#'
#' @param data  data
#' @param model  model
#' @param nStart  nStart
#' @param maxK  maxK
#'
#' @import cluster
#' @import factoextra
#' @import stats
#' @import ggplot2
#'
#' @export

clusteringVis <- function(data = NULL,
                          model = NULL,
                          nStart = 25,
                          maxK = 10){

  optimalK <- fviz_nbclust(data, stats::kmeans, method = "silhouette")

  clustVis <- factoextra::fviz_cluster(object = model,
                                       data = data,
                                       palette = c("#2E9FDF", "#00AFBB", "#E7B800"),
                                       geom = "point",
                                       ellipse.type = "convex",
                                       ggtheme = ggplot2::theme_bw()
  )

  return(list(clustVis = clustVis, optimalK = optimalK))
}
