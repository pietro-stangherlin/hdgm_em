# HDGM EM

## Summary

The goal of this project was to implement in C++ (with R interface using Rcpp and RcppArmadillo)
from scratch the EM algorithm to fit the spatio-temporal state space model (HDGM)
present in the paper of Otto et. al (2024) (https://link.springer.com/article/10.1007/s10651-023-00589-0)
from the agrimonia project about the daily monitoring of PM 2.5 along with some covariates
across many meterological stations in the north Italy Lombardy region from 2016 to 2020.

In the doc folder is present the html file abd its corresponding quadro source code with
some elementary theory and references about the methods used.

NOTE: in the R folder the agrimonia_preprocess.R script
is almost equal from the R code of the paper
and requires the agrimonia dataset (not included in this repo) which can be found at https://zenodo.org/records/7956006.

## Motivation

Since the article implemention of the EM its in Matlab I took the change to produce
something new and hopefully helpful while learning in the process.

In R there's, to my knowledge the MARSS package (https://cran.r-project.org/web/packages/MARSS) by
Holmes et. al, which is really great but has some drawbacks:

- it does not have an immediate (for what I've seen) implementaion of spatial covariance functions
in the state updates (like exponential or Matern correlations)
- it's mainly coded in R, which makes the fitting process quite slow compared to a C++ implementation

From the same authors of MARSS there's the package MARSTMB (https://atsa-es.github.io/marssTMB/)
which has the goal to implement C++ faster fitting of MARSS models, but at the time of the project
(and maybe still today) it was in beta and the last commit was older than 2 years old.

Of course this project code is a lot less flexible then what the MARSS package offers in terms 
of modelling possibilities.


## Some trivia

In order to not reinvent the wheel I've used as an initial template for Kalman filter and smoother
from "dfms: Dynamic Factor Models package" (https://cran.r-project.org/web/packages/dfms/index.html)
of Sebastian Krantz [aut, cre], Rytis Bagdziunas [aut], Santtu Tikka [rev], Eli Holmes [rev].


## Future developmens
See the html in doc.

