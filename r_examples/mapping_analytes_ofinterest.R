#####################################
######    IMPORTS NEEDED       ######
#####################################

library(sf)
library(tidyverse)
library(dplyr)
library(ggspatial)
library(cowplot)
library(ggplot2)
library(viridis)

#####################################
######    DATAFRAMES NEEDED    ######
#####################################

full_data_analytes_ofinterest_zip <- cleaned_median_50_noNA_filtered_NZVremoved_death_popprop_income %>% select(year, zip_code, total_deaths, total_population, radium_228, perchlorate, combined_uranium, foaming_agents__surfactants_, chloroform, nitrate_nitrite, alkalinity__carbonate, magnesium, X1_2_dibromo_3_chloropropane, dibromoacetic_acid)


zcta_shape <- st_read("C:/Users/kbk10/Desktop/Thesis_Analysis/SHP_Files/Corrected_California_Zip_Codes.shp")

zcta_shape <-  zcta_shape%>%
  rename(zip_code = ZIP_CODE) %>%
  mutate(zip_code = as.character(zip_code))

##########################################
######    DATAFRAME FOR ANALYSIS    ######
##########################################

# Summarize median concentration by ZIP for each analyte (replacing mean with median)
analyte_summary_med <- full_data_analytes_ofinterest_zip %>%
  group_by(zip_code) %>%
  dplyr::summarize(
    combined_uranium = median(combined_uranium, na.rm = TRUE),
    radium_228 = median(radium_228, na.rm = TRUE),
    chloroform = median(chloroform, na.rm = TRUE),
    nitrate_nitrite = median(nitrate_nitrite, na.rm = TRUE),
    X1_2_dibromo_3_chloropropane = median(X1_2_dibromo_3_chloropropane, na.rm = TRUE),
    dibromoacetic_acid = median(dibromoacetic_acid, na.rm = TRUE),
    perchlorate = median(perchlorate, na.rm = TRUE),
    foaming_agents__surfactants_ = median(foaming_agents__surfactants_, na.rm = TRUE),  
    alkalinity__carbonate = median(alkalinity__carbonate, na.rm = TRUE),
    magnesium = median(magnesium, na.rm = TRUE), 
  ) %>%
  mutate(zip_code = as.character(zip_code))

# Join the shape data with the analyte summary data
zip_map <- left_join(zcta_shape, analyte_summary_med, by = "zip_code")


##########################################
######            MAPPING           ######
##########################################



#####           MEDIAN         #####
# Create the individual ggplot objects
plot_uranium <- ggplot(zip_map) +
  geom_sf(aes(fill = combined_uranium), color = NA) +
  scale_fill_viridis_c(option = "C", na.value = "gray90") +
  labs(title = "Median Combined Uranium Concentration by ZIP Code",
       fill = "Concentration") +
  theme_minimal()

plot_uranium

# Create the individual ggplot objects
plot_nitrate_nitrite <- ggplot(zip_map) +
  geom_sf(aes(fill = nitrate_nitrite), color = NA) +
  scale_fill_viridis_c(option = "C", na.value = "gray90") +
  labs(title = "Median Nitrate/Nitrite Concentration by ZIP Code",
       fill = "Concentration") +
  theme_minimal()

plot_radium_228 <- ggplot(zip_map) +
  geom_sf(aes(fill = radium_228), color = NA) +
  scale_fill_viridis_c(option = "C", na.value = "gray90") +
  labs(title = "Median Radium 228 Concentration by ZIP Code",
       fill = "Concentration") +
  theme_minimal()

plot_chloroform <- ggplot(zip_map) +
  geom_sf(aes(fill = chloroform), color = NA) +
  scale_fill_viridis_c(option = "C", na.value = "gray90") +
  labs(title = "Median Chloroform Concentration by ZIP Code",
       fill = "Concentration") +
  theme_minimal()

plot_perchlorate <- ggplot(zip_map) +
  geom_sf(aes(fill = perchlorate), color = NA) +
  scale_fill_viridis_c(option = "C", na.value = "gray90") +
  labs(title = "Median Hardness Concentration by ZIP Code",
       fill = "Concentration") +
  theme_minimal()

plot_dibromo_3_chloropropane <- ggplot(zip_map) +
  geom_sf(aes(fill = X1_2_dibromo_3_chloropropane), color = NA) +
  scale_fill_viridis_c(option = "C", na.value = "gray90") +
  labs(title = "Median X1 2 Dibromo 3 Chloropropane Concentration by ZIP Code",
       fill = "Concentration") +
  theme_minimal()

plot_dibromoacetic_acid <- ggplot(zip_map) +
  geom_sf(aes(fill = dibromoacetic_acid), color = NA) +
  scale_fill_viridis_c(option = "C", na.value = "gray90") +
  labs(title = "Median Dibromoacetic Acid Concentration by ZIP Code",
       fill = "Concentration") +
  theme_minimal()

plot_magnesium <- ggplot(zip_map) +
  geom_sf(aes(fill = magnesium), color = NA) +
  scale_fill_viridis_c(option = "C", na.value = "gray90") +
  labs(title = "Median Hardness Concentration by ZIP Code",
       fill = "Concentration") +
  theme_minimal()

plot_foaming_agents <- ggplot(zip_map) +
  geom_sf(aes(fill = foaming_agents__surfactants_), color = NA) +
  scale_fill_viridis_c(option = "C", na.value = "gray90") +
  labs(title = "Median X1 2 Dibromo 3 Chloropropane Concentration by ZIP Code",
       fill = "Concentration") +
  theme_minimal()

plot_alkalinity <- ggplot(zip_map) +
  geom_sf(aes(fill = alkalinity__carbonate), color = NA) +
  scale_fill_viridis_c(option = "C", na.value = "gray90") +
  labs(title = "Median Dibromoacetic Acid Concentration by ZIP Code",
       fill = "Concentration") +
  theme_minimal()

# Combine the plots using cowplot
combined_plot <- plot_grid(
  plot_nitrate_nitrite, plot_radium_228, plot_chloroform, plot_alkalinity, plot_uranium,
  plot_dibromo_3_chloropropane, plot_dibromoacetic_acid, plot_magnesium, plot_foaming_agents, plot_perchlorate,
  ncol = 2, nrow = 5  # Adjust the number of columns and rows
)

# Print the combined plot
combined_plot



######################################################
####                                            #####
####          LAST 5 YEARS (TOPCODED)           ##### 
####                                            #####


# Filter to last 5 years
last_5_years <- full_data_analytes_ofinterest_zip %>%
  filter(year >= max(year, na.rm = TRUE) - 4)

# Topcode at 95th percentile
topcoded_data <- last_5_years %>%
  mutate(
    radium_228 = pmin(radium_228, quantile(radium_228, 0.95, na.rm = TRUE)),
    perchlorate = pmin(perchlorate, quantile(perchlorate, 0.95, na.rm = TRUE)),
    foaming_agents__surfactants_ = pmin(foaming_agents__surfactants_, quantile(foaming_agents__surfactants_, 0.95, na.rm = TRUE)),
    alkalinity__carbonate = pmin(alkalinity__carbonate, quantile(alkalinity__carbonate, 0.95, na.rm = TRUE)),
  )

# Median concentration by ZIP
analyte_summary_topcoded <- topcoded_data %>%
  group_by(zip_code) %>%
  dplyr::summarize(
    radium_228 = median(radium_228, na.rm = TRUE),
    perchlorate = median(perchlorate, na.rm = TRUE),
    foaming_agents__surfactants_ = median(foaming_agents__surfactants_, na.rm = TRUE),
    alkalinity__carbonate = median(alkalinity__carbonate, na.rm = TRUE)
  ) %>%
  mutate(zip_code = as.character(zip_code))

# Merge with shapefile
zip_map <- left_join(zcta_shape, analyte_summary_topcoded, by = "zip_code")



p1 <- ggplot(zip_map) +
  geom_sf(aes(fill = perchlorate), color = NA) +
  scale_fill_viridis_c(option = "C", na.value = "gray90") +
  labs(title = "Median Perchlorate Concentration 2017-2022", fill = "µg/L") +
  theme_void(base_size = 14)

p2 <- ggplot(zip_map) +
  geom_sf(aes(fill = radium_228), color = NA) +
  scale_fill_viridis_c(option = "C", na.value = "gray90") +
  labs(title = "Median Radium 228 Concentration 2017-2022", fill = "pCi/L") +
  theme_void(base_size = 14)

p3 <- ggplot(zip_map) +
  geom_sf(aes(fill = foaming_agents__surfactants_), color = NA) +
  scale_fill_viridis_c(option = "C", na.value = "gray90") +
  labs(title = "Median Foaming Agents Surfactants Concentration 2017-2022", fill = "µg/L") +
  theme_void(base_size = 14)

p4 <- ggplot(zip_map) +
  geom_sf(aes(fill = alkalinity__carbonate), color = NA) +
  scale_fill_viridis_c(option = "C", na.value = "gray90") +
  labs(title = "Median Alkalinity Carbonate Concentration 2017-2022", fill = "mg/L") +
  theme_void(base_size = 14)


# Adjust plot titles to just show the analyte name
p1 <- p1 + labs(title = "Perchlorate")
p2 <- p2 + labs(title = "Radium 228")
p3 <- p3 + labs(title = "Foaming Agents Surfactants")
p4 <- p4 + labs(title = "Alkalinity Carbonate")

# Combine the plots
combined_plot_map <- plot_grid(p1, p2, p3, p4, 
                               ncol = 2,
                               align = "hv")

# Stack title and combined plot vertically
####I USED THIS FORMATTING TO GET PLOTS IN 3X4 OFF CENTERED PLOT 
###final_plot_map <- plot_grid(title, combined_plot_map,
                            ##ncol = 1,
                            ##rel_heights = c(0.1, 1))  # Adjust the height of the title row

# Display final plot
###print(final_plot_map)

print(combined_plot_map)

