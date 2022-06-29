## Workflow example (k means clustering) ##
# Ref: https://uc-r.github.io/kmeans_clustering

## 유저로부터 받는 입력은 camel case,
##예시로 사용한 변수 및 snake case로 작성된 dependencies의 함수명은 snake case로 표기합니다.

## data import
library(tidyverse)
library(tidymodels)
library(dplyr)
library(recipes)
library(parsnip)
library(tune)
library(rsample)
library(vip)
library(ggrepel)
library(ggfortify)
library(ggdendro)
library(goophi)

library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization

url = 'http://www.biz.uiowa.edu/faculty/jledolter/DataMining/protein.csv'
mydata <- readr::read_csv(url)
data_cleaned <- mydata %>% dplyr::select(RedMeat, WhiteMeat, Eggs, Milk) ## unsupervised -> target 변수는 제외

## 여기까지 완료된 데이터가 전달된다고 가정 (one-hot encoding까지 되는지 확인 필요) ##

# user로부터 아래 정보를 입력받습니다
maxK <- 15 # k = 2:maxK
selectOptimal <- "silhouette" # "silhouette", "gap_stat" // there's no mathematical definition for selecting optimal k using elbow method.
nstart <- 25 # attempts 25 initial configurations

# K-means clustering 모델을 생성합니다.
km_model <- goophi::kMeansClustering_phi(data_cleaned,
                                         maxK = 10,
                                         nstart = 25,
                                         selectOptimal = "gap_stat",
                                         seed_num = 6471)

km_model
