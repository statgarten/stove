mode <- "classification"
algo <- "LogisticR"
engine <- "brulee"

penaltyRangeMin <- "0.5"
penaltyRangeMax <- "0.5"
penaltyRangeLevels <- "1"
mixtureRangeMin <- "0.5"
mixtureRangeMax <- "0.5"
mixtureRangeLevels <- "1"

v <- "2"

metric <- "roc_auc"


penaltyRange <- c(as.numeric(penaltyRangeMin), as.numeric(penaltyRangeMax))
mixtureRange <- c(as.double(mixtureRangeMin), as.double(mixtureRangeMax))

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


set.seed(seed = 1234)
tunedWorkflow <- workflows::workflow() %>%
  workflows::add_recipe(rec) %>%
  workflows::add_model(model)

gridSearchResult <- tune::tune_grid(tunedWorkflow,
                          resamples = rsample::vfold_cv(data_train, v = 2),
                          grid = parameterGrid
)

finalized <- goophi::fitBestModel(
  gridSearchResult = gridSearchResult,
  metric = metric,
  model = model,
  formula = formula,
  trainingData = data_train,
  splitedData = splitedData,
  algo = paste0(algo, "_", engine)
)


#################################################################




# mode <- "classification"
# algo <- "XGBoost"
# engine <- "xgboost"
#
# treeDepthRangeMin <- "5"
# treeDepthRangeMax <- "15"
# treeDepthRangeLevels <- "3"
# treesRangeMin <- "8"
# treesRangeMax <- "100"
# treesRangeLevels <- "3"
# learnRateRangeMin <- "-2"
# learnRateRangeMax <- "-1"
# learnRateRangeLevels <- "2"
# mtryRangeMin <- "0"
# mtryRangeMax <- "1"
# mtryRangeLevels <- "3"
# minNRangeMin <- "2"
# minNRangeMax <- "40"
# minNRangeLevels <- "3"
# lossReductionRangeMin <- "-1"
# lossReductionRangeMax <- "1"
# lossReductionRangeLevels <- "3"
# sampleSizeRangeMin <- "0.0"
# sampleSizeRangeMax <- "1.0"
# sampleSizeRangeLevels <- "3"
# stopIter <- "30"
#
# v <- "2"
#
# metric <- "roc_auc"
#
# treeDepthRange <- c(as.numeric(treeDepthRangeMin), as.numeric(treeDepthRangeMax))
# treesRange <- c(as.numeric(treesRangeMin), as.numeric(treesRangeMax))
# learnRateRange <- c(as.numeric(learnRateRangeMin), as.numeric(learnRateRangeMax))
# mtryRange <- c(as.numeric(mtryRangeMin), as.numeric(mtryRangeMax))
# minNRange <- c(as.numeric(minNRangeMin), as.numeric(minNRangeMax))
# lossReductionRange <- c(as.numeric(lossReductionRangeMin), as.numeric(lossReductionRangeMax))
# sampleSizeRange <- c(as.numeric(sampleSizeRangeMin), as.numeric(sampleSizeRangeMax))
# stopIterRange <- c(as.numeric(stopIter), as.numeric(stopIter))


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

set.seed(seed = 1234)
tunedWorkflow <- workflows::workflow() %>%
  workflows::add_recipe(rec) %>%
  workflows::add_model(model)

result <- tune::tune_grid(tunedWorkflow,
                          resamples = rsample::vfold_cv(data_train, v = 2),
                          grid = parameterGrid
)
