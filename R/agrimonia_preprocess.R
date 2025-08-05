BOOL_GENERAL_PREPROCESS = TRUE
BOOL_HDGM_EM_PREPROCESS = TRUE

if(BOOL_GENERAL_PREPROCESS){
  # script used to preprocess agrimonia dataset
  # both for GAMM and HDGM models used in agrimonia article

  # set working dir to project dir

  load("../offline_agrimonia/data/Agrimonia_Dataset_v_3_0_0.Rdata")
  agrim_df = AgrImOnIA_Dataset_v_3_0_0
  rm(AgrImOnIA_Dataset_v_3_0_0)

  # Deleting 2021
  agrim_df = agrim_df[agrim_df$Time < as.Date("2021/01/01"), ]

  # Delete switzerland stations
  "%notin%" = Negate("%in%")
  agrim_df = agrim_df[
    agrim_df$IDStations %notin%
      c("STA-CH0011A", "STA-CH0033A", "STA-CH0043A"),
  ]

  # include just stations measuring PM2.5 once
  stations_PM25 = unique(agrim_df$IDStations[!is.na(agrim_df$AQ_pm25)])
  agrim_df = agrim_df[agrim_df$IDStations %in% stations_PM25, ]

  # accorpate two stations that are actually one: STA.IT2282A & STA.IT1518A
  # (change occurred because it stopped measuring AQ_no2)
  # see differences between the two stations and accorpate them
  # they share equal values for covariates so we joint them
  agrim_df = agrim_df[
    (agrim_df$Time < as.Date("2019/01/01") &
       agrim_df$IDStations == "STA.IT2282A") == F,
  ]
  agrim_df$IDStations[
    agrim_df$IDStations == "STA.IT1518A" &
      agrim_df$Time < as.Date("2019/01/01")
  ] = "STA.IT2282A"

  agrim_df = agrim_df[agrim_df$IDStations != "STA.IT1518A", ]
  unique(agrim_df$Latitude[agrim_df$IDStations == "STA.IT2282A"])
  agrim_df$Latitude[agrim_df$IDStations == "STA.IT2282A"] = 45.4397
  unique(agrim_df$Longitude[agrim_df$IDStations == "STA.IT2282A"])
  agrim_df$Longitude[agrim_df$IDStations == "STA.IT2282A"] = 8.6204

  # we exclude rows with NA values
  agrim_df$AQ_pm25[agrim_df$AQ_pm25 < 0.5] = NA

  # Adding month
  agrim_df$Month = months(agrim_df$Time)
  agrim_df$Month = factor(agrim_df$Month, levels = c("gennaio", "febbraio", "marzo",
                                                     "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre"))

  # Remove rows with NAs in AQ_pm25
  agrim_df = agrim_df[!is.na(agrim_df$AQ_pm25), ]

  # Remove useless variables
  agrim_df = agrim_df |>
    dplyr::select(IDStations, Latitude, Longitude, Time, Altitude, AQ_pm25, WE_temp_2m,
                  WE_tot_precipitation, WE_rh_mean, WE_wind_speed_100m_mean, WE_blh_layer_max,
                  LI_pigs_v2, LI_bovine_v2, LA_hvi, LA_lvi, Month)

  save(agrim_df, file = "data/agri_df.RData")

  # Create a subsample with data from 2016 and 2017 only
  # agrim_df_subsmpl = agrim_df |>
  #   filter(Time < as.Date("2018/01/01"))

}

if(BOOL_HDGM_EM_PREPROCESS){
  source("R/preprocessing_helper.R")

  load("data/agri_df.RData")

  unique_times_sorted <- sort(unique(agrim_df[["Time"]]))
  N_times <- length(unique_times_sorted)

  # Compute distance between each station
  unique_map <- unique(agrim_df[,c("IDStations","Latitude","Longitude")])
  unique_stations <- unique_map[["IDStations"]]
  q <- length(unique_stations)

  coords <- unique_map[,c("Latitude","Longitude")]
  dists_matr <- as.matrix(dist(coords))

  # selected variables

  response_name <- "AQ_pm25"
  selected_vars_names <- c("Altitude","WE_temp_2m", "WE_tot_precipitation", "WE_rh_mean" ,
                           "WE_wind_speed_100m_mean", "WE_blh_layer_max", "LI_pigs_v2",
                           "LI_bovine_v2", "LA_hvi", "LA_lvi") # Month missing at the moment
  indicator_name = "IDStations"
  time_name <- "Time"

  # allocate response matrix and covariates array
  y.matr <- matrix(NaN, nrow = q, ncol = N_times)
  # add months
  X.array <- array(NaN, dim = c(q, length(selected_vars_names) + 12, N_times))

  # populate

  for(i in 1:N_times){
    col_index <- which(names(agrim_df) == time_name)

    temp_window <- agrim_df[agrim_df[[time_name]] == unique_times_sorted[i], -col_index]

    y.matr[,i] <- as.numeric(PermuteVector(
      as.numeric(temp_window[[response_name]]),
      temp_window[[indicator_name]],
      unique_stations))

    temp_matr <- as.matrix(temp_window[ , which(names(temp_window) %in% selected_vars_names)])
    # add month indicator variable with january as reference
    zeros_months_matr <- matrix(0, nrow = NROW(temp_matr), ncol = 11)
    month_index <- which(levels(agrim_df[["Month"]]) == base::months(unique_times_sorted[i]))
    if(month_index > 1){
      # excluding reference january
      zeros_months_matr[,month_index - 1] <- rep(1, NROW(temp_matr))
    }
    # adding intercept and months
    # january is the reference
    temp_matr <- cbind(rep(1,NROW(temp_matr)), zeros_months_matr, temp_matr)

    X.array[,,i] <- PermuteMatrix(
      temp_matr,
      temp_window[[indicator_name]],
      unique_stations
    )
  }

  save(y.matr, X.array, file = "data/agri_matrix_array_em.RData")

}
