# :yellow_heart: stove <img src="logo.png" width="120" align="right"/>
<!-- badges: start -->
[![Lifecycle:experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
<!-- badges: end -->
The stove package provides functions for ML modeling. [Packages from the Tidymodels](https://www.tidymodels.org/packages/) were used, but they were configured to be easy for ML beginners to use. Although it belongs to [statgarten](https://github.com/statgarten) whose packages are incorporated in shiny app, stove package also can be used for itself in console.

## :wrench: Install

``` r
# install.packages("devtools")
devtools::install_github("statgarten/stove")
```

## Example Code

### 1. Sample Data Import

```{r}
# remotes::install_github("statgarten/datatoys")
library(stove)
library(datatoys)
library(dplyr)

set.seed(1234)

cleaned_data <- datatoys::bloodTest

cleaned_data <- cleaned_data %>%
  mutate_at(vars(SEX, ANE, IHD, STK), factor) %>%
  mutate(TG = ifelse(TG < 150, 0, 1)) %>%
  mutate_at(vars(TG), factor) %>%
  group_by(TG) %>%
  sample_n(500) # TG(0):TG(1) = 500:500
```

### 2. Data split and Define preprocessing

```{r}
target_var <- "TG"
train_set_ratio <- 0.7
seed <- 1234
formula <- paste0(target_var, " ~ .")

# Split data

split_tmp <- stove::trainTestSplit(data = cleaned_data,
                                   target = target_var,
                                   prop = train_set_ratio,
                                   seed = seed
                                   )

data_train <- split_tmp[[1]] # train data
data_test <- split_tmp[[2]] # test data
data_split <- split_tmp[[3]] # whole data with split information

# Define preprocessing recipe for cross validation

rec <- stove::prepForCV(data = data_train,
                        formula = formula,
                        imputation = T,
                        normalization = T,
                        seed = seed
                        )
```

### 3. Modeling

```{r}
# User input

mode <- "classification"
algo <- "logisticRegression" # Custom name
engine <- "glmnet" # glmnet (default)
v <- 2
metric <- "roc_auc" # roc_auc (default), accuracy
gridNum <- 5
iter <- 10
seed <- 1234

# Modeling using logistic regression algorithm

finalized <- stove::logisticRegression(
  algo = algo,
  engine = engine,
  mode = mode,
  trainingData = data_train,
  splitedData = data_split,
  formula = formula,
  rec = rec,
  v = v,
  gridNum = gridNum,
  iter = iter,
  metric = metric,
  seed = seed
)
```
You can compare several models' performance and visualize them.  
These [documents](https://github.com/statgarten/stove/tree/main/quarto-doc) contain the example codes for modeling workflow using stove.


## :clipboard: Dependency

assertthat - 0.2.1\
base64enc - 0.1-3\
bayesplot - 1.10.0\
boot - 1.3-28.1\
C50 - 0.1.7\
callr - 3.7.3\
class - 7.3-20\
cli - 3.6.0\
cluster - 2.1.4\
codetools - 0.2-18\
colorspace - 2.0-3\
colourpicker - 1.2.0\
combinat - 0.0-8\
cowplot - 1.1.1\
crayon - 1.5.2\
crosstalk - 1.2.0\
Cubist - 0.4.1\
data.table - 1.14.6\
DBI - 1.1.3\
dials - 1.1.0\
DiceDesign - 1.9\
digest - 0.6.31\
discrim - 1.0.0\
dplyr - 1.0.10\
DT - 0.26\
dygraphs - 1.1.1.6\
ellipsis - 0.3.2\
factoextra - 1.0.7\
fansi - 1.0.3\
fastmap - 1.1.0\
forcats - 0.5.2\
foreach - 1.5.2\
Formula - 1.2-4\
furrr - 0.3.1\
future - 1.30.0\
future.apply - 1.10.0\
generics - 0.1.3\
ggplot2 - 3.4.0\
ggrepel - 0.9.2\
glmnet - 4.1-6\
globals - 0.16.2\
glue - 1.6.2\
gower - 1.0.1\
GPfit - 1.0-8\
gridExtra - 2.3\
gtable - 0.3.1\
gtools - 3.9.4\
hardhat - 1.2.0\
haven - 2.5.1\
highr - 0.1\
hms - 1.1.2\
htmltools - 0.5.4\
htmlwidgets - 1.6.1\
httpuv - 1.6.7\
igraph - 1.3.5\
inline - 0.3.19\
inum - 1.0-4\
ipred - 0.9-13\
iterators - 1.0.14\
kknn - 1.3.1\
klaR - 1.7-1\
labelled - 2.10.0\
later - 1.3.0\
lattice - 0.20-45\
lava - 1.7.1\
lhs - 1.1.6\
libcoin - 1.0-9\
lifecycle - 1.0.3\
listenv - 0.9.0\
lme4 - 1.1-31\
loo - 2.5.1\
lubridate - 1.9.0\
magrittr - 2.0.3\
markdown - 1.4\
MASS - 7.3-58.1\
Matrix - 1.5-3\
matrixStats - 0.63.0\
mime - 0.12\
miniUI - 0.1.1.1\
minqa - 1.2.5\
munsell - 0.5.0\
mvtnorm - 1.1-3\
naivebayes - 0.9.7\
nlme - 3.1-161\
nloptr - 2.0.3\
nnet - 7.3-18\
parallelly - 1.33.0\
parsnip - 1.0.3\
partykit - 1.2-16\
pillar - 1.8.1\
pkgbuild - 1.4.0\
pkgconfig - 2.0.3\
plyr - 1.8.8\
prettyunits - 1.1.1\
processx - 3.8.0\
prodlim - 2019.11.13\
promises - 1.2.0.1\
ps - 1.7.0\
purrr - 0.3.4\
questionr - 0.7.7\
R6 - 2.5.1\
randomForest - 4.7-1.1\
ranger - 0.14.1\
RColorBrewer - 1.1-3\
Rcpp - 1.0.9\
RcppParallel - 5.1.6\
recipes - 1.0.3\
reshape2 - 1.4.4\
rlang -\
rpart - 4.1.19\
rsample - 1.1.1\
rstan - 2.21.7\
rstanarm - 2.21.3\
rstantools - 2.2.0\
rstudioapi - 0.14\
scales - 1.2.1\
sessioninfo - 1.2.2\
shape - 1.4.6\
shiny - 1.7.4\
shinyjs - 2.1.0\
shinystan - 2.6.0\
shinythemes - 1.2.0\
StanHeaders - 2.21.0-7\
stringi - 1.7.8\
stringr - 1.5.0\
survival - 3.5-0\
threejs - 0.3.3\
tibble - 3.1.8\
tidyr - 1.2.1\
tidyselect - 1.2.0\
timechange - 0.1.1\
timeDate - 4022.108\
treesnip - 0.1.0.9001\
tune - 1.0.1\
utf8 - 1.2.2\
vctrs - 0.5.1\
withr - 2.5.0\
workflows - 1.1.2\
xtable - 1.8-4\
xts - 0.12.2\
yardstick - 1.1.0\
zoo - 1.8-11

## :blush: Authors

-   Yeonchan Seong [\@ycseong07](http://github.com/ycseong07)

## :memo: License

Copyright :copyright: 2022 Yeonchan Seong This project is [MIT](https://opensource.org/licenses/MIT) licensed
