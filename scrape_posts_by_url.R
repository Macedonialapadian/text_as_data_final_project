################################################################################
#                      Scrape Posts by URL from Nitter                         #
#                                                                              #
# Purpose: Collect individual posts by their tweet IDs/URLs                    #
################################################################################

# Domains to be used
nitter_domains <- c("https://nitter.tiekoetter.com",
                    "https://nitter.kareem.one")

################################################################################
#                               Function Definitions                           #
################################################################################

library(reticulate)
library(rvest)
library(purrr)
library(tidyverse)

# Required Libraries
seleniumbase <- import("seleniumbase")
SB <- seleniumbase$SB

################################################################################
#                              Read URLs from File                             #
################################################################################

read_urls_from_file <- function(file_path) {
  # Read file where each row has URLs/numbers separated by commas or spaces
  lines <- readLines(file_path)

  # Parse all URLs
  all_urls <- unlist(lapply(lines, function(line) {
    # Split by comma, space, or both
    urls <- strsplit(line, "[,\\s]+")[[1]]
    trimws(urls)
  }))

  # Remove empty strings
  all_urls <- all_urls[all_urls != ""]

  return(all_urls)
}

################################################################################
#                           Extract Single Post Data                           #
################################################################################

extract_single_post <- function(html) {
  # Find the main tweet (not replies)
  tweet <- html %>% html_element(".main-tweet .timeline-item, .main-tweet")

  if (is.na(tweet)) {
    tweet <- html %>% html_element(".timeline-item")
  }

  if (is.na(tweet)) {
    return(NULL)
  }

  # Extract stats
  stats <- html_elements(tweet, ".tweet-stat")
  spans <- html_elements(stats, "span")
  span_classes <- html_attr(spans, "class")

  extract_stat <- function(icon) {
    match_index <- which(str_detect(span_classes, icon))
    if (length(match_index)) {
      html_text(stats[[match_index[1]]]) %>% str_trim()
    } else {
      NA_character_
    }
  }

  tibble(
    datetime   = tweet %>% html_element(".tweet-date a") %>% html_attr("title"),
    content    = tweet %>% html_element(".tweet-content") %>% html_text(),
    likes      = extract_stat("icon-heart"),
    retweets   = extract_stat("icon-retweet"),
    comments   = extract_stat("icon-comment"),
    quotes     = extract_stat("icon-quote"),
    views      = extract_stat("icon-play")  # Views might use play icon or similar
  )
}

################################################################################
#                                 Helper Functions                             #
################################################################################

# Check if being rate limited. If so, refresh.
pass_cf_limiting <- function(sb, max_retries = 10){

  html <- sb$get_page_source()
  retries <- 0

  while(grepl("You are being rate limited", html) & retries <= max_retries){

    sb$refresh_page()
    sb$wait_for_ready_state_complete()
    Sys.sleep(2)
    retries <- retries + 1
    html <- sb$get_page_source()

  }

}

################################################################################
#                           Scrape Single Post Function                        #
################################################################################

scrape_single_post <- function(sb, tweet_url, domain = "https://nitter.tiekoetter.com") {

  # Construct full URL if only ID provided
  if (!grepl("^http", tweet_url)) {
    # Assume it's just a tweet ID, construct URL
    # Format: domain/i/status/TWEET_ID
    full_url <- paste0(domain, "/i/status/", tweet_url)
  } else {
    full_url <- tweet_url
  }

  # Navigate to the post
  sb$open(full_url)
  sb$wait_for_ready_state_complete()

  # Handle rate limiting
  pass_cf_limiting(sb)

  # Wait a moment for content to load
  Sys.sleep(1)

  # Get page source and parse
  html <- read_html(sb$get_page_source())

  # Extract post data
  post_data <- extract_single_post(html)

  if (!is.null(post_data)) {
    post_data$url <- tweet_url
  }

  return(post_data)
}

################################################################################
#                              Main Scraping Function                          #
################################################################################

scrape_posts_by_url <- function(urls_file,
                                 output_file = "scraped_posts.csv",
                                 domain = "https://nitter.tiekoetter.com",
                                 headless = FALSE) {

  # Read URLs from file
  urls <- read_urls_from_file(urls_file)
  message(paste("Found", length(urls), "URLs to scrape"))

  #####################################
  # Initialize browser

  ctx <- SB(uc = TRUE, test = FALSE, locale = "en", headless = headless)
  sb <- ctx$`__enter__`()
  sb$activate_cdp_mode(domain)
  sb$wait_for_ready_state_complete()
  sb$uc_gui_click_captcha()

  sb$wait_for_ready_state_complete()
  sb$open(paste0(domain, "/settings"))

  sb$wait_for_ready_state_complete()
  sb$select_if_unselected('input[name="infiniteScroll"]')
  sb$unselect_if_selected('input[name="mp4Playback"]')
  sb$unselect_if_selected('input[name="proxyVideos"]')
  sb$unselect_if_selected('input[name="autoplayGifs"]')

  sb$click('button.pref-submit')
  Sys.sleep(1)

  #####################################
  # Scrape each URL

  all_posts <- list()

  for (i in seq_along(urls)) {

    tryCatch({

      message(paste("Scraping URL", i, "of", length(urls), ":", urls[i]))

      post_data <- scrape_single_post(sb, urls[i], domain)

      if (!is.null(post_data)) {
        all_posts[[length(all_posts) + 1]] <- post_data
      }

      # Small delay between requests
      Sys.sleep(runif(1, 0.5, 1.5))

    }, error = function(e) {
      message(paste("Error scraping URL", urls[i], ":", e$message))
    })

  }

  # Exit browser
  ctx$`__exit__`(NULL, NULL, NULL)

  #####################################
  # Combine and clean data

  if (length(all_posts) > 0) {

    result <- bind_rows(all_posts)

    # Parse datetime
    result$datetime <- parse_date_time(gsub(" · ", " ", result$datetime),
                                       orders = "b d, Y I:M p", tz = "UTC")

    # Convert stats to numeric
    result$likes <- as.numeric(gsub(",", "", result$likes))
    result$retweets <- as.numeric(gsub(",", "", result$retweets))
    result$comments <- as.numeric(gsub(",", "", result$comments))
    result$quotes <- as.numeric(gsub(",", "", result$quotes))
    result$views <- as.numeric(gsub(",", "", result$views))

    # Reorder columns
    result <- result %>%
      select(url, datetime, content, likes, retweets, comments, quotes, views)

    # Save to CSV
    write.csv(result, output_file, row.names = FALSE)
    message(paste("Saved", nrow(result), "posts to", output_file))

    return(result)

  } else {

    message("No posts were successfully scraped")
    return(NULL)

  }

}

################################################################################
#                                  Example Usage                               #
################################################################################

# Example: Create a sample URLs file
# Each line can have multiple tweet IDs separated by commas
#
# Example urls.txt content:
# 1234567890123456789,1234567890123456790,1234567890123456791
# 1234567890123456792,1234567890123456793
# 1234567890123456794

# Run the scraper
# result <- scrape_posts_by_url(
#   urls_file = "urls.txt",
#   output_file = "scraped_posts.csv",
#   domain = nitter_domains[1],
#   headless = TRUE
# )

# Or with visible browser for debugging
# result <- scrape_posts_by_url(
#   urls_file = "urls.txt",
#   output_file = "scraped_posts.csv",
#   domain = nitter_domains[1],
#   headless = FALSE
# )
