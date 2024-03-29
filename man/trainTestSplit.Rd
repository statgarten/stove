% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/preprocessing.R
\name{trainTestSplit}
\alias{trainTestSplit}
\title{Train-Test Split}
\usage{
trainTestSplit(data = NULL, target = NULL, prop, seed = "4814")
}
\arguments{
\item{data}{Full data set with global preprocess completed.}

\item{target}{The target variable.}

\item{prop}{Proportion of total data to be used as training data.}

\item{seed}{Seed for reproducible results.}
}
\description{
Train-Test Split
}
\details{
Separate the entire data into a training set and a test set.
}
