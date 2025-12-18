# How Do U.S. Politicians Use Emotional Intensity?

**Course:** Text as Data  
**Student:** Chris Cheng

---

## 📌 Project Overview
This research investigates the relationship between the emotional intensity of U.S. politicians' language on social media and the resulting levels of public engagement.

### Motivation
The intuition for this project stems from a comparative observation of social media platforms. I observed that content on Twitter (X) often feels significantly more extreme in its emotional delivery compared to platforms like Weibo. 

### Key Findings
* **Intensity Drives Engagement:** There is a direct positive correlation between the emotional intensity of a post and public interaction metrics (likes, comments, views, and retweets). 
* **Partisan Differences:** Compared to Democrats, Republicans tend to use less intense language on average, but the public shows a higher sensitivity (stronger reaction) to intense language when used by Republican politicians.
* **Topic-Specific Intensity:** Using **Structural Topic Modeling (STM)**, the analysis reveals that politicians are more likely to use intense language on less controversial or "consensus" topics (e.g., "get out the vote" campaigns or sharing emergency information like hurricane updates), while remaining relatively more cautious on highly controversial political topics.

---

## 📊 Methodology
* **Language Labeling:** We utilized a **RoBERTa model** to assign intensity scores to posts.
* **Measurement:** Language intensity was measured using the **Best-Worst Scaling (BWS)** method.
* **Topic Analysis:** **Structural Topic Modeling (STM)** was employed to analyze the distribution of intensity across various political topics.

---

## 📂 Repository Structure

* **`documents/`**: Contains the project proposal and the final research report.
* **`models/`**: (External) Includes the RoBERTa model used for intensity labeling and the STM model files.
* **`plots/`**: Contains all visualizations, including the STM analysis plots.
* **`data/`**: (External) This folder contains the raw and processed data (see below).

---

## 💾 Data and Model Access
Due to the size of the datasets, the `data/` and `model/`folder is not included in this repository. 

**[Download the Data and Model Folder Here](https://drive.google.com/drive/folders/1FrU9BMDN_WdOttR-Yo43gEWN7K6TuB6C?usp=sharing)**

**Installation:** Download the folder and place it in the **root path** of this project directory.
The data folder contains:
* `/cleaned`: Data processed and ready for analysis.
* `/scraper_material_data`: Files organized and prepared for the scraping process.
* `/scraper_result_data`: Raw output retrieved from the scraper.

---

## ⚙️ Replication Instructions
To replicate this research, please run the scripts in the root according to their numerical prefix:

1.  **Main Scripts (e.g., `01_...`, `02_...`):** Run these in numerical order to produce the final results.
2.  **Assistance Scripts (e.g., `01a_...`, `01b_...`):** These scripts handle supplementary tasks or data cleaning. They are helpful for understanding the workflow but are not the primary drivers of the final output.
