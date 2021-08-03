# Online Shopping Review Analysis using Topic Modeling ########################
# A. PROJECT DETAILS ##########################################################

## Dataset: Online Shopping Review
## Date   : April 12th 2021
## Email  : amirudin.adi@gmail.com

# B. DATA PREPARATION #########################################################

# 1. Importing Dataset & Library ##############################################

## 1.1. Importing Package Library #############################################

### Basic Helpful Packages:
library(dplyr)
library(tidyr)
library(caret)
library(broom)
library(ggplot2)
library(psych)
library(tidyverse)

### NLP Packages:
library(topicmodels)
library(NLP)
library(tm)
library(readr)
library(tidytext)
library(stringr)

### Graphics & Data Mapping:
library(leaflet)
library(gridExtra)
library(plotly)
library(highcharter)
library(wordcloud)

## 1.2. Importing Dataset #####################################################
revdata <- read.csv("D:/Data Analysis/Personal Projects Directory/Topic Modeling - Online Shopping Review/Reviews.csv")
View(revdata)
revdata <- as_tibble(revdata)
head(revdata)
dim(revdata)
summary(revdata)
glimpse(revdata)

# 2. Data Cleaning Part I #####################################################

## 2.1. Missing Data Treatment ################################################
sapply(revdata,function(x) {sum(is.na(x),na.rm=TRUE)/length(x)}) #Checking for missing
sum(is.na(revdata)) #found 4 missing value
mean(is.na(revdata)) #mean is 0.000001137
revdata <- na.omit(revdata) #missing value treatment

## 2.2. Drop Unused Data ######################################################
revdata <- revdata[ ,-c(1, 8)] # delete columns Id and Time

# 3. Data Conversion/Wrangling ################################################

## 3.1. Convert Score to Factor ###############################################
revdata <- revdata %>% mutate_if(is.numeric, as.factor)
revdata2 %>% mutate_if(is.numeric, as.factor) #Convert numeric to factor
glimpse(revdata)

## 3.2. Creating Text Tibble ##################################################
revsumm <- paste(revdata$Summary,collapse = " ")
revtext <- paste0(revdata$Text, collapse = " ")

# C. TEXT PREPARATION #########################################################

# 4. Setting Up Source and Corpus #############################################

## 4.1. Source and Corpus for Text ############################################
rev_source <- VectorSource(revtext) 
corpus     <- Corpus(rev_source)

### 4.1.2. Change to Lowercase ################################################
corpus <- tm_map(corpus,content_transformer(tolower))
#Translate characters in character vectors, in particular from upper to lower case or vice versa.

### 4.1.3. Removing Punctuation ###############################################
corpus <- tm_map(corpus,content_transformer(removePunctuation))
#Remove punctuation marks from a text document.

### 4.1.4. Removing Numbers Content ###########################################
corpus <- tm_map(corpus,content_transformer(removeNumbers))
#Remove numbers from a text document.

### 4.1.5. Whitespace Stripping ###############################################
corpus <- tm_map(corpus,stripWhitespace)
#Strip extra whitespace from a text document. Multiple whitespace characters are collapsed to a single blank.

### 4.1.6. Removing Stopwords #################################################
corpus <- tm_map(corpus,removeWords,stopwords(kind="en"))
#Removing stopwords

### 4.1.7. Creating Document term Matrix ######################################
dtm <-DocumentTermMatrix(corpus)
dtmm <-as.matrix(dtm)

## 4.2. Source and Corpus for Title ###########################################
rev_source2 <- VectorSource(revsumm) 
corpus2     <- Corpus(rev_source2)

### 4.2.2. Change to Lowercase ################################################
corpus2 <- tm_map2(corpus2,content_transformer(tolower))
#Translate characters in character vectors, in particular from upper to lower case or vice versa.

### 4.2.3. Removing Punctuation ###############################################
corpus2 <- tm_map(corpus2,content_transformer(removePunctuation))
#Remove punctuation marks from a text document.

### 4.2.4. Removing Numbers Content ###########################################
corpus2 <- tm_map(corpus2,content_transformer(removeNumbers))
#Remove numbers from a text document.

### 4.2.5. Whitespace Stripping ###############################################
corpus2 <- tm_map(corpus2,stripWhitespace)
#Strip extra whitespace from a text document. Multiple whitespace characters are collapsed to a single blank.

### 4.2.6. Removing Stopwords #################################################
corpus2 <- tm_map(corpus2,removeWords,stopwords(kind="en"))
#Removing stopwords

### 4.2.7. Creating Document term Matrix ######################################
dtm2 <-DocumentTermMatrix(corpus2)
dtmm2 <-as.matrix(dtm2)

# D. TEXT ANALYSIS ############################################################
# 5. Explanatory Data Analysis ################################################
## 5.1. Finding The Most Frequent Words #######################################
freq_term <-colSums(dtmm)
freq_term <-sort(freq_term,decreasing = TRUE)
words <-names(freq_term)

freq_term2 <-colSums(dtmm2)
freq_term2 <-sort(freq_term,decreasing = TRUE)
words2 <-names(freq_term2)

## 5.2. Word Cloud Plotting ###################################################
wordcloud(words[1:200],freq_term[1:200],random.order = FALSE,colors=brewer.pal(8,"Dark2"))

wordcloud(words2[1:200],freq_term2[1:200],random.order = FALSE,colors=brewer.pal(8,"Dark2"))

 