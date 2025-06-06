% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/methods.R
\name{residuals.dfm}
\alias{residuals.dfm}
\alias{resid.dfm}
\alias{fitted.dfm}
\title{DFM Residuals and Fitted Values}
\usage{
\method{residuals}{dfm}(
  object,
  method = switch(object$em.method, none = "2s", "qml"),
  orig.format = FALSE,
  standardized = FALSE,
  na.keep = TRUE,
  ...
)

\method{fitted}{dfm}(
  object,
  method = switch(object$em.method, none = "2s", "qml"),
  orig.format = FALSE,
  standardized = FALSE,
  na.keep = TRUE,
  ...
)
}
\arguments{
\item{object}{an object of class 'dfm'.}

\item{method}{character. The factor estimates to use: one of \code{"qml"}, \code{"2s"} or \code{"pca"}.}

\item{orig.format}{logical. \code{TRUE} returns residuals/fitted values in a data format similar to \code{X}.}

\item{standardized}{logical. \code{FALSE} will put residuals/fitted values on the original data scale.}

\item{na.keep}{logical. \code{TRUE} inserts missing values where \code{X} is missing (default \code{TRUE} as residuals/fitted values are only defined for observed data). \code{FALSE} returns the raw prediction, which can be used to interpolate data based on the DFM. For residuals, \code{FALSE} returns the difference between the prediction and the initial imputed version of \code{X} use for PCA to initialize the Kalman Filter.}

\item{\dots}{not used.}
}
\value{
A matrix of DFM residuals or fitted values. If \code{orig.format = TRUE} the format may be different, e.g. a data frame.
}
\description{
The residuals \eqn{\textbf{e}_t = \textbf{x}_t - \textbf{C} \textbf{F}_t}{e(t) = x(t) - C F(t)} or fitted values \eqn{\textbf{C} \textbf{F}_t}{C F(t)} of the DFM observation equation.
}
\examples{
\donttest{
library(xts)
# Fit DFM with 3 factors and 3 lags in the transition equation
mod = DFM(diff(BM14_M), r = 3, p = 3)

# Residuals
head(resid(mod))
plot(resid(mod, orig.format = TRUE)) # this is an xts object

# Fitted values
head(fitted(mod))
head(fitted(mod, orig.format = TRUE)) # this is an xts object
}

}
\seealso{
\link{dfms-package}
}
