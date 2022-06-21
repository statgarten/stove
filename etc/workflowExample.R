## https://github.com/rahul-raoniar/Rahul_Raoniar_Blogs/tree/main/Modeling%20Logistic%20Regression%20using%20Tidymodels%20Library%20in%20R

library(mlbench)
library(tidymodels)
library(tibble)

#### import data ####

## data frame to tibble
data(PimaIndiansDiabetes2)
PimaIndiansDiabetes2 <- tibble::as_tibble(PimaIndiansDiabetes2)

## view the structures of data
glimpse(PimaIndiansDiabetes2)
str(PimaIndiansDiabetes2)

#### data preprocessing ####

## removing NA values
Diabetes <- na.omit(PimaIndiansDiabetes2)
glimpse(Diabetes)

## check the levels of outcome
levels(Diabetes$diabetes)

## setting reference level
Diabetes$diabetes <- relevel(Diabetes$diabetes, ref = "pos")
levels(Diabetes$diabetes)

## Train-Test Split
set.seed(123)

diabetes_split <- initial_split(Diabetes,
                                prop = 0.75,
                                strata = diabetes)

diabetes_train <- diabetes_split %>%
  training()

diabetes_test <- diabetes_split %>%
  testing()

nrow(diabetes_train)
nrow(diabetes_test)


## Cross validation (추가예정정)


## fitting logistic regression
# fitted_logistic_model<- logistic_reg() %>%
#   set_engine("glm") %>%
#   set_mode("classification") %>%
#   fit(diabetes~., data = diabetes_train)

f <- "diabetes~."
fitted_logistic_model <- goophi::LogisticRegression(data = diabetes_train, formula = f)

## result

tidy(fitted_logistic_model)

tidy(fitted_logistic_model, exponentiate = TRUE)

tidy(fitted_logistic_model, exponentiate = TRUE) %>%
  filter(p.value < 0.05)

## class prediction
pred_class <- predict(fitted_logistic_model,
                      new_data = diabetes_test,
                      type = "class")

pred_class[1:5,]

## Prediction Probabilities
pred_proba <- predict(fitted_logistic_model,
                      new_data = diabetes_test,
                      type = "prob")

## both
diabetes_results <- diabetes_test %>%
  select(diabetes) %>%
  bind_cols(pred_class, pred_proba)

diabetes_results[1:5, ]


## confusion matrix
conf_mat(diabetes_results, truth = diabetes,
         estimate = .pred_class)


## accuracy
accuracy(diabetes_results, truth = diabetes,
         estimate = .pred_class)

## sensitivity (Sensitivity = TP / FN+TP) == Recall
sens(diabetes_results, truth = diabetes,
     estimate = .pred_class)

recall(diabetes_results, truth = diabetes,
       estimate = .pred_class)

## specificity (Specificity = TN/FP+TN.)
spec(diabetes_results, truth = diabetes,
     estimate = .pred_class)

## precision (Precision = TP/TP+FP)
precision(diabetes_results, truth = diabetes,
          estimate = .pred_class)

## F1 score
f_meas(diabetes_results, truth = diabetes,
       estimate = .pred_class)

## kappa
kap(diabetes_results, truth = diabetes,
    estimate = .pred_class)

## Matthews Correlation Coefficient (MCC)
mcc(diabetes_results, truth = diabetes,
    estimate = .pred_class)

## combined reuslt
custom_metrics <- metric_set(accuracy, sens, spec, precision, recall, f_meas, kap, mcc)
custom_metrics(diabetes_results,
               truth = diabetes,
               estimate = .pred_class)

## AUROC
roc_auc(diabetes_results,
        truth = diabetes,
        .pred_pos)

## ROC curve
diabetes_results %>%
  roc_curve(truth = diabetes, .pred_pos) %>%
  autoplot()
