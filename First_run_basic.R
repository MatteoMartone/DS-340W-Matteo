# Matteo Martone
# DS 340W
# Sample Code 1


packages <- c("readxl", "tidyverse", "knitr", "kableExtra", "hasseDiagram", "parameters", "emmeans", "DescTools","dunn.test", "multcompView", "effectsize", "psych")

lapply(packages, library, character.only = TRUE)

# Loading Helper Files and Setting Global Options
source("https://raw.github.com/neilhatfield/STAT461/master/rScripts/ANOVATools.R")
options("contrasts" = c("contr.sum", "contr.poly"))

# Tell Knitr to use empty space instead of NA in printed tables
options(knitr.kable.NA = "")

# Load Additional Tools ----
source("https://raw.github.com/neilhatfield/STAT461/master/rScripts/ANOVATools.R")
source("https://raw.github.com/neilhatfield/STAT461/master/rScripts/shadowgram.R")

#read in 
combine13 <- read_excel("NFL 2013_edit.xlsx")
combine14 <- read_excel("NFL 2014_edit.xlsx")
combine15 <- read_excel("NFL 2015_edit.xlsx")
combine16 <- read_excel("NFL 2016_edit.xlsx")
combine17 <- read_excel("NFL 2017_edit.xlsx")

#summary stats
summary_combine13 <- summary(combine13)
summary_combine14 <- summary(combine14)
summary_combine15 <- summary(combine15)
summary_combine16 <- summary(combine16)
summary_combine17 <- summary(combine17)

#ex summary 
print(summary_combine13)

# Add 'Year' column in combine17
combine17$Year <- 2017
combine17 <- combine17[, c("Year", colnames(combine17)[1:(ncol(combine17)-1)])]

# Remove '10 Yard' column from combine16 if it exists
combine16 <- subset(combine16, select = -`10 Yard`)

# Standardize column names across all datasets
colnames(combine13) <- c("Year", "Name", "Pos", "College", "Height", "Weight", "40yd", "BP", "Vertical", "Broad Jump", "Shuttle", "3Cone")
colnames(combine14) <- c("Year", "Name", "Pos", "College", "Height", "Weight", "40yd", "BP", "Vertical", "Broad Jump", "Shuttle", "3Cone")
colnames(combine15) <- c("Year", "Name", "Pos", "College", "Height", "Weight", "40yd", "BP", "Vertical", "Broad Jump", "Shuttle", "3Cone")
colnames(combine16) <- c("Year", "Name", "Pos", "College", "Height", "Weight", "40yd", "BP", "Vertical", "Broad Jump", "Shuttle", "3Cone")
colnames(combine17) <- c("Year", "Name", "Pos", "College", "Height", "Weight", "40yd", "BP", "Vertical", "Broad Jump", "Shuttle", "3Cone")

# Combine all datasets into one
combined_data <- rbind(combine13, combine14, combine15, combine16, combine17)

# View the first few rows of the combined dataset
head(combined_data)

#summary
summary_stats <- summary(combined_data)
print(summary_stats)

# 40-yard dash time by position
ggplot(combined_data, aes(x = Pos, y = `40yd`)) +
  geom_boxplot() +
  labs(title = "40-yard Dash Time by Position", x = "Position", y = "40-yard Dash Time (seconds)") +
  theme_minimal()

# Vertical jump by position
ggplot(combined_data, aes(x = Pos, y = Vertical)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "Vertical Jump by Position", x = "Position", y = "Vertical Jump (inches)") +
  theme_minimal()

# Weight distribution by position
ggplot(combined_data, aes(x = Pos, y = Weight)) +
  geom_boxplot(fill = "lightgreen") +
  labs(title = "Weight Distribution by Position", x = "Position", y = "Weight (lbs)") +
  theme_minimal()
