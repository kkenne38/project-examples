##############################
####    IMPORTS NEEDED    ####
library(dplyr)
library(fixest)
library(tidyr)
library(ggplot2)
library(caret)
library(Hmisc)


################################################################################
#                         RUNNING FE MODEL                              

###Dataframe needed 
df_final_adjustments_wo_outliers <- read_csv("df_final_adjustments_wo_outliers.csv", col_names = TRUE)

############################################
####         SCALING DATASET           #####
############################################

# List of columns to exclude
exclude_cols <- c("year", "zip_code", "under_5_pct", "x5_14_pct", "x15_24_pct", "x25_64_pct", "x65_plus_pct", "total_population", "total_deaths", "median_income", "GW_binary")

# Identify all columns except the excluded ones
analyte_cols <- setdiff(names(df_final_adjustments_wo_outliers), exclude_cols)

# Check the result
print(analyte_cols)


complete_scaled_cleaned_median_adjusted_PWS <- df_final_adjustments_wo_outliers %>%
  mutate(across(all_of(analyte_cols), ~ scale(.)[, 1]))

#########################################################
####   RUNNING 2-WAY FIXED EFFECTS REGRESSION       #####
####     WITH CLUSTERED SE AND BH CORRECTION        ##### 
#########################################################

# Initialize an error tracker
error_vars <- c()

# Define the model-fitting function (clustered SE)
fit_model_for_column <- function(data_5, response_var_5, predictor_var_5) {
  tryCatch({
    predictor_var_quoted <- paste0("`", predictor_var_5, "`")
    
    controls <- c("under_5_pct", "x5_14_pct", "x15_24_pct", "x25_64_pct", "x65_plus_pct", "median_income", "GW_binary")
    controls_quoted <- paste0("`", controls, "`", collapse = " + ")
    formula_string <- paste(response_var_5, "~", predictor_var_quoted, "+", controls_quoted)
    formula <- as.formula(formula_string)
    
    poisson_model <- feglm(
      fml = formula, 
      data = data_5,
      family = poisson(),
      offset = log(data_5[['total_population']]),
      fixef = c("year", "zip_code")
    )
    
    vcov_clustered <- ~ zip_code
    coefficient_table_5 <- coeftable(poisson_model, vcov = vcov_clustered)
    CI <- confint(poisson_model, vcov = vcov_clustered)
    
    if (!(predictor_var_5 %in% rownames(coefficient_table_5))) {
      stop(paste("Predictor", predictor_var_5, "not found in coefficient table."))
    }
    
    results_5 <- list(
      Predictor = predictor_var_5,
      Coefficient = exp(coefficient_table_5[predictor_var_5, "Estimate"]),
      Std_Error = coefficient_table_5[predictor_var_5, "Std. Error"],
      z_value = coefficient_table_5[predictor_var_5, "z value"],
      p_value = coefficient_table_5[predictor_var_5, "Pr(>|z|)"],
      CI_Lower = exp(CI[predictor_var_5, 1]),
      CI_Upper = exp(CI[predictor_var_5, 2])
    )
    return(results_5)
  }, error = function(e) {
    print(paste("Error with", predictor_var_5, ":", e$message))
    error_vars <<- c(error_vars, predictor_var_5)
    return(NULL)
  })
}

# Run clustered SE models
data_5 <- complete_scaled_cleaned_median_adjusted_PWS
predictor_vars_5 <- setdiff(names(data_5), c("year", "zip_code", "under_5_pct", "x5_14_pct", "x15_24_pct", "x25_64_pct", "x65_plus_pct", "total_deaths", "total_population", "median_income", "GW_binary"))
response_var_5 <- "total_deaths"
models_results_5 <- list()
for (predictor in predictor_vars_5) {
  models_results_5[[predictor]] <- fit_model_for_column(data_5, response_var_5, predictor)
}
all_results_clustered <- do.call(rbind, lapply(models_results_5, as.data.frame))
all_results_clustered$BH_p <- p.adjust(all_results_clustered$p_value, method = "BH")
significant_results_BH_clustered_adjusted_popincomesource <- all_results_clustered %>%
  filter(p_value < 0.05) %>%
  arrange(p_value)
significant_results_BH_clustered_adjusted_popincomesource <- all_results_clustered %>%
  filter(BH_p < 0.05) %>%
  arrange(BH_p)

write.csv(significant_results_BH_clustered_adjusted_popincomesource, "significant_results_BH_clustered_adjusted_popincomesource.csv", row.names = TRUE)


##################################################
####         CREATING TABLE OF  RESULTS     #####
#################################################

generateAnalyteTable <- function(models_results_5) {
  
  # Extract all raw p-values to compute BH-adjusted p-values
  raw_pvalues <- sapply(models_results_5, function(res) res$p_value)
  bh_pvalues <- p.adjust(raw_pvalues, method = "BH")
  
  # Apply getGLMResults logic to each model and include BH p-values
  analyte_table <- do.call(rbind, lapply(seq_along(models_results_5), function(i) {
    res <- models_results_5[[i]]
    
    c <- log(res$Coefficient)
    std.err <- res$Std_Error
    CI <- (exp(std.err * qnorm(0.975)) - 1) * 100
    cExp <- (exp(c) - 1) * 100 
    lower <- (exp(c - 1.96 * std.err) - 1) * 100
    upper <- (exp(c + 1.96 * std.err) - 1) * 100
    pValue <- res$p_value
    bh_p <- bh_pvalues[i]
    
    rr <- sqrt(exp(c))
    if (rr >= 1) {
      EValue <- rr + sqrt(rr * (rr - 1))
    } else {
      EValue <- (1 / rr) + sqrt((1 / rr) * ((1 / rr) - 1))
    }
    
    # Return row
    data.frame(
      Analyte = res$Predictor,
      Coefficient = round(c, 4),
      Std_Err = round(std.err, 3),
      Increase = round(cExp, 3),
      Lower = round(lower, 3),
      Upper = round(upper, 3),
      P_Value = ifelse(pValue == 0, "<0.001", round(pValue, 3)),
      BH_P = ifelse(bh_p < 0.001, "<0.001", round(bh_p, 3)),
      E_Value = round(EValue, 3)
    )
  }))
  
  return(analyte_table)
}

analyte_table <- generateAnalyteTable(models_results_5)
write.csv(analyte_table, "analyte_regression_table2.csv", row.names = FALSE)

analyte_regression_table <- analyte_table %>%
  mutate(Analyte = recode(Analyte, "Adicarb Sulfone" = "Aldicarb Sulfone"))

library(dplyr)
library(gt)

combined_table <- analyte_regression_table %>%
  left_join(mds_df_cat, by = c("Analyte" = "Analyte_pretty")) %>%
  arrange(Category, BH_P) %>%
  mutate(
    `Increase (95% CI)` = sprintf("%.2f (%.2f–%.2f)", Increase, Lower, Upper)
  ) %>%
  select(
    Category,
    Analyte,
    Coefficient,
    Std_Err,
    `Increase (95% CI)`,
    P_Value,
    BH_P,
    E_Value
  )

final_table <- combined_table %>%
  gt(groupname_col = "Category", rowname_col = "Analyte") %>%
  cols_label(
    Coefficient = "Coefficient",
    Std_Err = "SE",
    `Increase (95% CI)` = "% Increase (95% CI)",
    BH_P = html("<i>P</i> value (BH-adjusted)"),
    E_Value = "E-value"
  ) %>%
  fmt_number(
    columns = where(is.numeric),
    decimals = 3
  ) %>%
  tab_header(
    title = "Table 1. Regression Results for Analyte-Associated Mortality Risk"
  ) %>%
  tab_footnote(
    footnote = "Regressions were conducted separately for each analyte, where year and zip-code were included as fixed effects and population proportions, median income, and water source were included as adjustment variables. Analyte concentration was calculated per zip-code as the annual weighted median concentration of all PWS measurments within each zip-code area. Increase is interpreted as the percent change in mortality per 1-unit (SD) increase in contaminant concentration. Coefficient references the regression coefficient, while SE refers to the SE. BH-adjusted p-values refer to Benjamini-Hochberg adjusted p-values to minimize type 1 errors of false discovery. E-value is the minimum strength of association that an unmeasured confounder would need to explain away the association.",
    locations = cells_title(groups = "title")
  ) %>%
  cols_align(
    align = "left",
    columns = everything()
  ) %>%
  tab_options(
    table.background.color = "#F5F5E9"
  ) %>%
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_body(rows = BH_P <= 0.05)
  ) %>%
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_stub(rows = BH_P <= 0.05)
  ) %>%
  tab_style(
    style = list(
      cell_fill(color = "#F7F7F5"),
      cell_text(weight = "bold")
      ),
    locations = cells_row_groups()
  )

gtsave(final_table, "regression_table.png", vwidth = 1000, vheight = 800)
write_xlsx(combined_table, "regression_table.xlsx")


############################################
####         PLOTTING RESULTS          #####
############################################

#FOREST 
ggplot(significant_results_BH_clustered_adjusted_popincomesource, aes(x = reorder(Predictor, Coefficient), y = Coefficient)) +
  geom_point() +
  geom_errorbar(aes(ymin = CI_Lower, ymax = CI_Upper), width = 0.2) +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Significant Regression Coefficients and 95% Confidence Intervals",
    x = "Analyte",
    y = "Coefficient"
  )

