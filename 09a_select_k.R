library(tidyverse)
library(stm)
library(lubridate)
library(quanteda)
library(parallel) # For detecting cores

# --- 1. Load and Clean Data (Same as 09_stm_pool.R) ---

# Load data
dta <- read_csv("data/cleaned/the_dataset.csv")
officer_data <- read_csv("data/raw/official_data.csv")

# Join and filter dates
dta$month <- month(dta$datetime)
dta <- left_join(dta, officer_data, by = "official_id")

dta <- dta %>% 
  filter(!month %in% c(1, 2, 3, 4)) # Remove Jan-April

dta$year <- year(dta$datetime)
dta <- dta %>% 
  filter(year == 2024) # Keep only 2024

# Recode Factors
dta <- dta %>%
  mutate(
    race = case_when(
      race == "White" ~ "White",
      race == "Black" ~ "Black",
      race == "Latino" ~ "Latino",
      race == "Asian American" ~ "Asian American",
      TRUE ~ "Other"
    ),
    race = factor(race, levels = c("White", "Black","Latino","Asian American","Other")),
    
    party = case_when(
      party == "Democratic" ~ "Democratic",
      party == "Republican" ~ "Republican",
      TRUE ~ "Other"
    ),
    party = factor(party, levels = c("Democratic", "Republican", "Other"))
  )

# --- 2. Pool Data (Same as 09_stm_pool.R) ---

dta_pooled <- dta %>%
  group_by(official_id, calendar_week) %>%
  summarise(
    content = paste(content, collapse = " "),
    predicted_bws_score = mean(predicted_bws_score, na.rm = TRUE),
    datetime = min(datetime),
    race = race,
    party = party,
    tweet_count = n(),
    .groups = "drop"
  ) %>%
  mutate(date_numeric = as.numeric(as.Date(datetime)))

# --- 3. Text Processing (Same as 09_stm_pool.R) ---

corp <- corpus(dta_pooled, text_field = "content")
docvars(corp) <- dta_pooled

my_stopwords <- c(
  stopwords("en"),
  stopwords("es"),
  "it’s", "i’m", "don’t", "can’t", "won’t",
  "just", "like",
  "go", "went", "going", "gone",
  "make", "made",
  "get", "got"
)

toks <- tokens(corp, 
               remove_punct = TRUE, 
               remove_symbols = FALSE, 
               remove_numbers = TRUE, 
               remove_url = TRUE) %>% 
  tokens_tolower() %>%
  tokens_remove(my_stopwords)

dfm_counts <- dfm(toks)
dfm_trimmed <- dfm_trim(dfm_counts, min_docfreq = 3)

out <- convert(dfm_trimmed, to = "stm")

# Create the race_party_combo metadata just in case you need it later
out$meta$race_party_combo <- paste(out$meta$party, out$meta$race, sep = "_")
out$meta$race_party_combo <- gsub(" ", "", out$meta$race_party_combo)

# --- 4. Search K (Topic Selection) ---

# Define the range of topics to test
# You can adjust this list, e.g., seq(10, 50, by = 5)
candidate_k <- c(12, 13,14, 15,16, 17,18,19,20,21,22)

message("Starting searchK... this may take some time.")

# Run searchK using the SAME prevalence formula as your final model
select <- searchK(
  documents = out$documents,
  vocab = out$vocab,
  K = candidate_k,
  prevalence = ~ s(predicted_bws_score) + s(date_numeric) + race + party,
  data = out$meta,
  init.type = "Spectral",
  seed = 114514,
  verbose = TRUE,
  cores = detectCores() - 1 # Uses all cores except one to keep computer responsive
)

# Save the heavy computation result immediately
saveRDS(select, "models/R_topic_model/select_k_results.rds")

# --- 5. Plotting Results ---

# Create a dataframe for ggplot
k_metrics <- data.frame(
  K = unlist(select$results$K),
  SemanticCoherence = unlist(select$results$semcoh),
  Exclusivity = unlist(select$results$exclus),
  Residuals = unlist(select$results$residual)
)

# Plot Semantic Coherence vs Exclusivity
p <- ggplot(k_metrics, aes(x = SemanticCoherence, y = Exclusivity, label = K)) +
  geom_point(size = 3, color = "darkblue", alpha = 0.7) +
  geom_text(vjust = -0.8, fontface = "bold") +
  theme_minimal() +
  labs(
    title = "Model Selection: Exclusivity vs. Semantic Coherence",
    subtitle = "Optimal K is typically towards the top-right corner",
    x = "Semantic Coherence (Higher is better)",
    y = "Exclusivity (Higher is better)"
  )

ggsave("models/R_topic_model/k_selection_plot.png", plot = p, width = 8, height = 6)

print(p)
