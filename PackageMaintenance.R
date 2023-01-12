# 필요 패키지 없는 경우 설치 및 로드

if (! ("devtools" %in% rownames(installed.packages()))) { install.packages("devtools") }
base::require("devtools")

if (! ("roxygen2" %in% rownames(installed.packages()))) { install.packages("roxygen2") }
base::require("roxygen2")

if (! ("testthat" %in% rownames(installed.packages()))) { install.packages("testthat") }
base::require("testthat")

if (! ("knitr" %in% rownames(installed.packages()))) { install.packages("knitr") }
base::require("knitr")

if (! ("quarto" %in% rownames(installed.packages()))) { install.packages("quarto") }
base::require("quarto")

# DESCRIPTION 파일에 패키지 추가
usethis::use_package("vetiver", type = "Imports")
usethis::use_package("readr", type = "Suggests")
usethis::use_dev_package("treesnip", remote = "https://github.com/curso-r/treesnip.git") # remotes

# /vignettes 폴더에 패키지 설명서 추가
usethis::use_vignette("clusteringWorkflow")

# 패키지에 데이터가 추가될 경우, 아래 함수로 해당 object를 /data 폴더에 저장
usethis::use_data()

# 환경변수 저장 파일 관리
usethis::edit_r_environ()

# R 파일 테스트
usethis::use_test()

# 함수 추가 시 roxygen 주석을 포함시켜 작성하고, 아래 코드로 주석을 .Rd 파일로 전환 및 NAMESPACE에 추가
devtools::document()

# quarto 문서 관리
quarto::quarto_preview("document.qmd")
quarto::quarto_render("document.qmd")

# pkgdown 작성
usethis::use_pkgdown()
pkgdown::build_site()
