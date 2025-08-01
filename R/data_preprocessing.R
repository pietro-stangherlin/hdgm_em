# preprocessing functions
# to convert data to pass to EM function

# Assumptions:

# there are T response observations
# each observation is a (q \times 1) vector y_t
# with an associated covariates matrix X_t (q \times p)

# 1) the order in the y_t vector counts:
# assuming there are q distinct locations with a distance metric defined between them

# 2) there can be missing data, if so for each y_t
# a full (q \times 1) vector is made with Nan values in missing values indexes
# and a similar thing is done for each X_t matrix where each missing index row
# is set to Nan

#' @param input_vec
#' @param sorting_variable_vals
#' @param correct_sorting_vec
PermuteVector <- function(input_vec,
                          sorting_variable_vals,
                          correct_sorting_vec){
  q = length(correct_sorting_vec)
  res_vec = rep(NaN, q)

  for(i in 1:length(input_vec)){
    temp_index = which(correct_sorting_vec == sorting_variable_vals[i])
    res_vec[temp_index] = input_vec[q]
  }

  return(res_vec)

}

#' @param input_mat
#' @param sorting_variable_vals
#' @param correct_sorting_vec
PermuteMatrix <- function(input_mat,
              sorting_variable_vals,
              correct_sorting_vec){
  p = NCOL(input_mat)
  q = length(correct_sorting_vec)

  res_mat = matrix(NaN, nrow = q, ncol = p)

  for(i in 1:NROW(input_mat)){
    temp_index <- which(correct_sorting_vec == sorting_variable_vals[i])
    res_mat[temp_index,] <- input_mat[i,]
  }

  return(res_mat)

}
