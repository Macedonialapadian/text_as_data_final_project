library(tidyverse)
library(stm)
library(lubridate)
library(quanteda)
library(parallel)

dta <- read_csv("data/cleaned/the_dataset.csv")
officer_data <- read_csv("data/raw/official_data.csv")

dta$month <- month(dta$datetime)
dta <- left_join(dta, officer_data, by="official_id")

dta <- dta %>% 
  filter(!month %in% c(1, 2, 3, 4))

dta$year <- year(dta$datetime)
dta <- dta %>% 
  filter(year == 2024)

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

dta_pooled <- dta %>%
  group_by(official_id, calendar_week) %>%
  summarise(
    content = paste(content, collapse = " "),
    predicted_bws_score = mean(predicted_bws_score, na.rm = TRUE),
    datetime = min(datetime),
    race = first(race),
    party = first(party),
    tweet_count = n(),
    .groups = "drop"
  ) %>%
  mutate(date_numeric = as.numeric(as.Date(datetime)))


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

out$meta$race_party_combo <- paste(out$meta$party, out$meta$race, sep = "_")
out$meta$race_party_combo <- gsub(" ", "", out$meta$race_party_combo)

topic_model <- stm(
  documents = out$documents, 
  vocab = out$vocab, 
  K = 16, 
  prevalence = ~ s(predicted_bws_score) + s(date_numeric)+race_party_combo,
  data = out$meta,
  init.type = "Spectral",
  seed = 114514,
  verbose = TRUE
)

estimate_model <- estimateEffect(
  formula = ~ s(predicted_bws_score) + s(date_numeric)+race_party_combo, 
  stmobj = topic_model, 
  meta = out$meta, 
  uncertainty = "Global"
)

dir.create("models/R_topic_model", recursive = TRUE, showWarnings = FALSE)
saveRDS(topic_model, "models/R_topic_model/stm_topic_model_16_pool.rds")
saveRDS(estimate_model, "models/R_topic_model/stm_topic_model_estimated_16_pool.rds")

dir.create("models/inputs", recursive = TRUE, showWarnings = FALSE)
saveRDS(out, "models/R_topic_model/inputs/out_pool.rds")
saveRDS(dta, "models/R_topic_model/inputs/dta_pool.rds")