library(readr)
library(dplyr)
library(lubridate)
library(ggplot2)

# -----------------------------
# 1. Load Data
# -----------------------------
files <- paste0("water_tables/SDWIS", 1:3, ".tab")

read_one_file <- function(file) {
  read_tsv(
    file,
    locale = locale(encoding = "ISO-8859-1"),
    col_types = cols_only(
      `Water System Number` = col_character(),
      `Population Served`   = col_double(),
      `Sampling Point Name` = col_character(),
      `Sample Date`         = col_character(),
      `Analyte Name`        = col_character(),
      `Result`              = col_double(),
      `Counting Error`      = col_double(),
      `Units of Measure`    = col_character(),
      `Less Than Reporting Level` = col_logical(),
      `Reporting Level`     = col_double(),
      DLR                   = col_double(),
      MCL                   = col_double()
    )
  )
}

water <- bind_rows(lapply(files, read_one_file)) %>%
  mutate(
    `Sample Date` = mdy(`Sample Date`),
    Sample_Year   = year(`Sample Date`)
  ) %>%
  arrange(`Sample Date`)

# -----------------------------
# 2. Clean Result Column
# -----------------------------
water <- water %>%
  mutate(
    Result = as.numeric(Result),
    Result = if_else(is.na(Result), `Reporting Level` / sqrt(2), Result)
  )

# -----------------------------
# 3. Fill in Less Than Reporting Level
# -----------------------------
water <- water %>%
  mutate(`Less Than Reporting Level` = case_when(
    is.na(`Less Than Reporting Level`) & Result <  `Reporting Level` ~ "Y",
    is.na(`Less Than Reporting Level`) & Result >  `Reporting Level` ~ "N",
    is.na(`Less Than Reporting Level`) & Result == `Reporting Level` ~ "Neither",
    TRUE ~ as.character(`Less Than Reporting Level`)
  ))

# -----------------------------
# 4. Unit Fix Function
# -----------------------------
fix_units <- function(df, analyte, default_unit, conversion_factor = NULL) {
  condition <- df$`Analyte Name` == analyte & df$`Units of Measure` == "MG/L"
  condition[is.na(condition)] <- FALSE
  
  if (!is.null(conversion_factor)) {
    df[condition, c("Reporting Level", "Result", "DLR", "MCL")] <-
      df[condition, c("Reporting Level", "Result", "DLR", "MCL")] * conversion_factor
    df$`Units of Measure`[condition] <- default_unit
  }
  
  # Fill blanks
  blank_condition <- df$`Analyte Name` == analyte & df$`Units of Measure` == ""
  df$`Units of Measure`[blank_condition] <- default_unit
  
  df
}

# Analyte-specific fixes
analyte_unit_fixes <- list(
  "CHLOROFORM"       = list(default_unit = "UG/L"),
  "LEAD"             = list(default_unit = "UG/L", conversion_factor = 1000),
  "COPPER, FREE"     = list(default_unit = "UG/L", conversion_factor = 1000),
  "NITRITE"          = list(default_unit = "UG/L", conversion_factor = 1000),
  "NITRATE-NITRITE"  = list(default_unit = "UG/L", conversion_factor = 1000),
  "AGGRESSIVE INDEX" = list(default_unit = "AGGR")
)

for (analyte in names(analyte_unit_fixes)) {
  params <- analyte_unit_fixes[[analyte]]
  water  <- fix_units(water, analyte, params$default_unit, params$conversion_factor)
}

# -----------------------------
# 5. Remove Impossible Values
# -----------------------------
water <- water %>%
  filter(!(tolower(`Analyte Name`) == "ph" & Result > 14))

# -----------------------------
# 6. Handle Missing Units by Imputation
# -----------------------------
water_updated <- water
unit_conversion_counts <- data.frame(Analyte = character(), Converted_Count = integer())

analytes_with_missing_units <- water_updated %>%
  filter(is.na(`Units of Measure`)) %>%
  distinct(`Analyte Name`) %>%
  pull()

for (analyte in analytes_with_missing_units) {
  
  common_units <- water_updated %>%
    filter(`Analyte Name` == analyte, !is.na(`Units of Measure`)) %>%
    group_by(`Sampling Point Name`, `Units of Measure`) %>%
    tally() %>%
    slice_max(n, with_ties = FALSE) %>%
    select(`Sampling Point Name`, most_common_unit = `Units of Measure`)
  
  rows_before <- water_updated %>%
    filter(`Analyte Name` == analyte & is.na(`Units of Measure`)) %>%
    nrow()
  
  water_updated <- water_updated %>%
    left_join(common_units, by = "Sampling Point Name") %>%
    mutate(`Units of Measure` = if_else(
      is.na(`Units of Measure`) & `Analyte Name` == analyte,
      most_common_unit,
      `Units of Measure`
    )) %>%
    select(-most_common_unit)
  
  rows_after <- water_updated %>%
    filter(`Analyte Name` == analyte & !is.na(`Units of Measure`)) %>%
    nrow() - (nrow(water_updated) - rows_before)
  
  unit_conversion_counts <- rbind(
    unit_conversion_counts,
    data.frame(Analyte = analyte, Converted_Count = rows_after)
  )
}

print(unit_conversion_counts)

# -----------------------------
# 7. Verify Unit Conversions
# -----------------------------
original_data <- water %>%
  filter(`Analyte Name` %in% names(analyte_unit_fixes)) %>%
  mutate(Result_Before = Result) %>%
  select(`Water System Number`, `Sample Date`, `Analyte Name`,
         `Units of Measure`, Result_Before)

updated_data <- water_updated %>%
  filter(`Analyte Name` %in% names(analyte_unit_fixes)) %>%
  mutate(Result_After = Result) %>%
  select(`Water System Number`, `Sample Date`, `Analyte Name`,
         `Units of Measure`, Result_After)

check_conversion <- original_data %>%
  inner_join(updated_data,
             by = c("Water System Number", "Sample Date", "Analyte Name"),
             suffix = c("_orig", "_updated")) %>%
  filter(`Units of Measure_orig` == "MG/L" & `Units of Measure_updated` != "MG/L") %>%
  select(`Analyte Name`, `Units of Measure_orig`, `Units of Measure_updated`,
         Result_Before, Result_After)

print(head(check_conversion, 20))

# -----------------------------
# 8. Diagnostics & Plots
# -----------------------------
# NA counts
analyte_nan <- water %>%
  group_by(`Analyte Name`) %>%
  summarise(NaN_Count = sum(is.na(Result)), .groups = "drop") %>%
  arrange(desc(NaN_Count))

ggplot(analyte_nan, aes(x = reorder(`Analyte Name`, -NaN_Count), y = NaN_Count)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(x = "Analyte", y = "Number of NA Values", title = "NA Counts by Analyte") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Unique Result counts
analyte_unique_counts <- water %>%
  group_by(`Analyte Name`) %>%
  summarise(Unique_Result_Count = n_distinct(Result, na.rm = TRUE), .groups = "drop") %>%
  arrange(desc(Unique_Result_Count))

ggplot(analyte_unique_counts, aes(x = reorder(`Analyte Name`, -Unique_Result_Count), y = Unique_Result_Count)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(x = "Analyte", y = "Unique Result Count", title = "Unique Results per Analyte") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Chloroform histogram (post-cleaning)
chloroform_data <- water_updated %>%
  filter(`Analyte Name` == "CHLOROFORM", !is.na(Result))

ggplot(chloroform_data, aes(x = Result)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
  labs(title = "Chloroform Result Distribution", x = "Chloroform Result", y = "Frequency") +
  theme_minimal()
