% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/fitting.R
\name{bayesOptCV}
\alias{bayesOptCV}
\title{Bayesian optimization with cross validation}
\usage{
bayesOptCV(
  rec = NULL,
  model = NULL,
  v = NULL,
  trainingData = NULL,
  gridNum = NULL,
  iter = NULL,
  seed = NULL
)
}
\arguments{
\item{rec}{The recipe object including local preprocessing.}

\item{model}{The model object including the list of hyperparameters, engine and mode.}

\item{v}{Perform cross-validation by dividing the training data into v folds.}

\item{trainingData}{The training data.}

\item{gridNum}{Initial number of iterations to run before starting the optimization algorithm.}

\item{iter}{The maximum number of search iterations.}

\item{seed}{Seed for reproducible results.}
}
\description{
Bayesian optimization with cross validation
}
\details{
Optimize the hyperparameters of the model with Cross Validation and Bayesian optimization.
}
