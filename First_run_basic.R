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

combine13 <- read_excel("NFL 2013_edit.xlsx")
combine14 <- read_excel("NFL 2014_edit.xlsx")
combine15 <- read_excel("NFL 2015_edit.xlsx")
combine16 <- read_excel("NFL 2016_edit.xlsx")
combine17 <- read_excel("NFL 2017_edit.xlsx")
