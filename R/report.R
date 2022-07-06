#' AUC-ROC Curve
#'
#' @details
#' AUC-ROC Curve
#'
#' @param models_list  models_list
#' @param targetVar  targetVar
#'
#' @import yardstick
#' @import RColorBrewer
#' @import dplyr
#'
#' @export

rocCurve <- function(models_list, targetVar){
  colors = RColorBrewer::brewer.pal(7, "Set2")[1:8] # maximum 8 models?

  plot <- do.call(rbind, models_list)[[5]] %>% ## rbind here does nothing
    do.call(rbind, .) %>%
    dplyr::group_by(model) %>%
    yardstick::roc_curve(truth = eval(parse(text = targetVar)),
                         .pred_1,
                         event_level = 'second') %>%
    ggplot(
      aes(
        x = 1-specificity,
        y = sensitivity,
        color = model
      )
    ) +
    labs(title = "ROC curve",
         x = "False Positive Rate (1-Specificity)",
         y = "True Positive Rate (Sensitivity)") +
    geom_line(size = 1.1) +
    geom_abline(slope = 1, intercept = 0, size = 0.5) +
    scale_color_manual(values = colors) +
    coord_fixed() +
    theme_cowplot()

  return(plot)
}


#' Confusion matrix
#'
#' @details
#' Confusion matrix
#'
#' @param i  i
#' @param models_list  models_list
#' @param targetVar  targetVar
#'
#' @import yardstick
#' @import tune
#' @import ggplot2
#'
#' @export

confusionMatrix <- function(i, models_list, targetVar){

  plot <- models_list[[as.numeric(i)]] %>%
    tune::collect_predictions() %>%
    yardstick::conf_mat(eval(parse(text = targetVar)), .pred_class) %>%
    ggplot2::autoplot(type = "heatmap") +
    ggplot2::labs(title = models_list[[as.numeric(i)]][[5]][[1]]$model[1])

  return(plot)
}


#' Evaluation metrics
#'
#' @details
#' Evaluation metrics
#'
#' @param i  i
#' @param models_list  models_list
#' @param targetVar  targetVar
#'
#' @import yardstick
#' @import tune
#' @import ggplot2
#'
#' @export

evalMetrics <- function(models_list, targetVar){

  table <- data.frame()
  custom_metrics <- yardstick::metric_set(yardstick::accuracy,
                                          yardstick::sens,
                                          yardstick::spec,
                                          yardstick::precision,
                                          yardstick::f_meas,
                                          yardstick::kap,
                                          yardstick::mcc
  )

  for (i in 1:length(models_list)) {
    tmp <- custom_metrics(models_list[[as.numeric(i)]] %>%
                          tune::collect_predictions(),
                        truth = eval(parse(text = targetVar)),
                        estimate = .pred_class) %>%
      dplyr::select(.estimate) %>%
      data.table::transpose() %>%
      dplyr::mutate(across(where(is.numeric), ~ round(., 3)))

    table <- rbind(table, tmp)
    rownames(table)[i] <- models_list[[as.numeric(i)]][[5]][[1]]$model[1]
  }
  colnames(table) <- c("Accuracy", "Recall", "Specificity", "Precision", "F1-score", "Kappa", "MCC")

  return(table)
}
