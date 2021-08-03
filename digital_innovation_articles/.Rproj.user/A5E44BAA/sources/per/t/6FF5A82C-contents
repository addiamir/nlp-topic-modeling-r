# Digital Innovation - Text Analysis & Topic Modeling with LDA ################
# A. PROJECT DETAILS ##########################################################

## Dataset: Digital Innovation Scientific Research
## Date   : April 16th 2021
## Email  : amirudin.adi@gmail.com

## 0. Question Lists ##########################################################
## Q1. Who is the most productive author? Hendriffson & No author name
## Q2. How is the trend of this topic? Growing in 2010~
## Q3. DIstribution in Language Used? English 92%
## Q4. What is the most prominent publisher? Answered
## Q5. What topic is the most frequent?
## Q6. What topic is the most cited?
## Q7. Relation between topics?
## Q8. 


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
library(lubridate)

### NLP Packages:
library(topicmodels)
library(NLP)
library(tm)
library(readr)
library(tidytext)
library(stringr)
library(lda)
library(textmineR)

### Graphics & Data Plotting:
library(leaflet)
library(gridExtra)
library(plotly)
library(highcharter)
library(wordcloud)
library(LDAvis)

## 1.2. Importing Dataset #####################################################
scidata <- read.csv("D:/Data Analysis/Personal Projects Directory/Topic Modeling - Digital Innovation Scientific Articles/DI_Dataset (Cleaned).csv")
scidata <- as_tibble(scidata)
View(scidata)
head(scidata)
dim(scidata)

# 2. Data Cleaning ############################################################

## 2.1. Removing Excessive Rows ###############################################
scidata <- scidata[ ,-c(49 : 1646)]
glimpse(scidata)

## 2.2. Renaming Variables ####################################################
scidata <- scidata %>% 
  rename(Authors = ï..Authors, Authors.ID = Author.s..ID, Art.No = Art..No.)
glimpse(scidata)

## 2.3. Selecting Used Data ###################################################
scidata <- scidata[ , c("Authors", "Authors.ID", "Title", "Year", "Source.title", 
                        "Cited.by", "Affiliations", "Authors.with.affiliations",
                        "Abstract", "Author.Keywords", "Index.Keywords", "Publisher", 
                        "Access.Type", "Language.of.Original.Document")]

## 2.5. Missing Data Treatment ################################################
sapply(scidata,function(x) {sum(is.na(x),na.rm=TRUE)/length(x)}) #Checking for missing
sum(is.na(scidata)) #found 355 missing value
mean(is.na(scidata)) #mean 0.03088568 

# 3. Data Conversion ##########################################################

## 3.1. Converting Numeric to Date ############################################
is.Date(scidata$Year)
scidata <- transform(scidata, Year = as.Date(as.character(Year), "%Y"))
is.Date(scidata$Year)
head(scidata$Year)

## 3.2. Converting Numeric to Factors #########################################

### 3.2.1. Variable: Access.Type ##############################################
scidata$Access.Type <- as.factor(scidata$Access.Type)
levels(scidata$Access.Type)[1] <- 'Other Access' #Assigning N/A factor to Others
is.factor(scidata$Access.Type)

### 3.2.2. Variable: Language.of.Original.Document ############################
scidata$Language.of.Original.Document <- as.factor(scidata$Language.of.Original.Document)
levels(scidata$Language.of.Original.Document)[1] <- '(Other)' #Assigning N/A factor to Others
is.factor(scidata$Language.of.Original.Document)

### 3.2.3. Variable: Publisher ################################################
scidata$Publisher <- as.factor(scidata$Publisher)
levels(scidata$Publisher)[1] <- '(Other)' #Assigning N/A factor to Others
is.factor(scidata$Publisher)

### 3.2.4. Variable: Authors ##################################################
yearauth <- scidata %>% 
  select(c('Year','Authors')) %>% 
  gather(key = 'role', value = 'person', Authors) %>% 
  filter(person != "") %>% separate_rows(person, sep = ',')
yearauth$person <- trimws(yearauth$person)
head(yearauth) #2069 authors contributed to this field of research

## 3.3. Converting Text Data ##################################################

### 3.3.1. Variable: Keywords from Author #####################################
authlist <- paste(scidata$Author.Keywords, collapse = " ", sep = ",")
corpusauth <- Corpus(VectorSource(authlist)) # make a corpus object
corpusauth <- tm_map(corpusauth,content_transformer(removePunctuation)) #clean corpus
dtmauth <- DocumentTermMatrix(corpusauth) # get the count of words/document
unique_indexes1 <- unique(dtmauth$i) # get the index of each unique value
dtmauth <- dtmauth[unique_indexes1,]#Remove any empty rows in our document term matrix (if there are any we'll get an error when we try to run our LDA)
dtmmauth <-as.matrix(dtmauth) #create document matrix

### 3.3.2. Variable: Keywords from Index ######################################
indlist <- paste(scidata$Index.Keywords, collapse = " ", sep = ",")
corpusind <- Corpus(VectorSource(indlist)) # make a corpus object
corpusind <- tm_map(corpusind,content_transformer(removePunctuation)) #clean corpus
dtmind <- DocumentTermMatrix(corpusind) # get the count of words/document
unique_indexes2 <- unique(dtmind$i)
dtmind <- dtmind[unique_indexes2,] # Remove any empty rows in our document term matrix (if there are any we'll get an error when we try to run our LDA)
dtmmind <-as.matrix(dtmind) #create document matrix

### 3.3.3. Variable: Title ####################################################
dtmtit <- CreateDtm(scidata$Title, 
                 doc_names = scidata$Title, 
                 ngram_window = c(1, 2),
                 stopword_vec = c(stopwords::stopwords("en"),
                                  stopwords::stopwords(source = "smart")), #remove stopwords
                 lower = TRUE, #lowercase
                 remove_punctuation = TRUE, 
                 remove_numbers = TRUE,
                 stem_lemma_function = NULL, #not stemming words because it causes headache (words lost meaning)
                 verbose = TRUE) #creating formal class document term matrix
dim(dtmtit) #check dimensions
tf_tit <- TermDocFreq(dtm = dtmtit)
vocabularytit <- tf_tit$term[ tf_tit$term_freq > 1 & tf_tit$doc_freq < nrow(dtmtit) / 2 ]
dtmtit <- dtmtit[ , vocabularytit]
dim(dtmtit) #check dimensions after vocabulary reduction

### 3.3.4. Variable: Abstract #################################################

#### 3.3.4.1. Unigram #########################################################
dtmABS <- CreateDtm(scidata$Abstract, 
                    doc_names = scidata$Title, 
                    ngram_window = c(1, 1),
                    stopword_vec = c(stopwords::stopwords("en"),
                                     stopwords::stopwords(source = "smart")), 
                    lower = TRUE,
                    remove_punctuation = TRUE, 
                    remove_numbers = TRUE,
                    stem_lemma_function = NULL, verbose = TRUE) #creating formal class document term matrix
dim(dtmABS) #check dimensions
tf_ABS <- TermDocFreq(dtm = dtmABS)
vocabularyABS <- tf_ABS$term[ tf_ABS$term_freq > 1 & tf_ABS$doc_freq < nrow(dtmABS) / 2 ]
dtmABS <- dtmABS[ , vocabularyABS]
dim(dtmABS) #check dimensions after vocabulary reduction


#### 3.3.4.2. Bi-Gram #########################################################
dtmABSBI <- CreateDtm(scidata$Abstract, 
                    doc_names = scidata$Title, 
                    ngram_window = c(1, 2),
                    stopword_vec = c(stopwords::stopwords("en"),
                                     stopwords::stopwords(source = "smart")), 
                    lower = TRUE,
                    remove_punctuation = TRUE, 
                    remove_numbers = TRUE,
                    stem_lemma_function = NULL, verbose = TRUE) #creating formal class document term matrix
dim(dtmABSBI) #check dimensions
tf_ABSBI <- TermDocFreq(dtm = dtmABSBI)
vocabularyABSBI <- tf_ABSBI$term[ tf_ABSBI$term_freq > 1 & tf_ABSBI$doc_freq < nrow(dtmABSBI) / 2 ]
dtmABSBI <- dtmABSBI[ , vocabularyABSBI]
dim(dtmABSBI) #check dimensions after vocabulary reduction

# C. DATA ANALYSIS ############################################################

# 4. Explanatory Data Analysis ################################################

## 4.1. No of Digital Innovation Research - Yearly ############################
scidata %>%
  group_by(Year) %>%
  summarise(Total=n()) %>%
  hchart(type="column",hcaes(x= Year,y=Total)) %>%
  hc_title(text="No of Digital Innovation Research-Yearly") %>%
  hc_add_theme(hc_theme_flatdark())

## 4.2. Digital Innovation Research Popular Publishers ########################
scidata %>%
  group_by(Publisher) %>%
  summarize(Total=n()) %>%
  hchart(type="column",hcaes(x=Publisher, y= Total)) %>%
  hc_add_theme(hc_theme_google()) %>%
  hc_title(text="Digital Innovation Research Popular Publishers")

## 4.3. Total Citations Throughout Years ######################################
scidata %>% 
  select(Cited.by) %>%
  filter(!is.na(Cited.by)) %>%
  group_by(Cited.by) %>%
  top_n(20) %>%
  summarise(Count = n()) %>%
  arrange(Count) %>%
  plot_ly(
    x = ~ Count ,
    y = ~ Cited.by,
    type = "bar",
    orientation = "h"
  ) %>%
  layout(yaxis = list(categoryorder = "array", categoryarray = ~ Count)) %>%
  layout(
    title = "Total Citation",
    yaxis = list(title = "Citation"),
    xaxis = list(title = "Frequency")
  )

## 4.4. Most Cited Articles ###################################################
scidata %>%
  group_by(Cited.by, Title) %>%
  summarize(Total=n()) %>%
  top_n(30) %>%
  hchart(type = "column",hcaes(x=Title, y=Cited.by)) %>%
  hc_add_theme(hc_theme_google()) %>%
  hc_title(text="The Most Cited Articles")

## 4.5. Top Contributors ######################################################
yearauth %>%
  group_by(yearauth$person) %>%
  summarize(Total=n()) %>%
  top_n(30) %>%
  hchart(type = "column",hcaes(x=yearauth$person, y= yearauth$Year)) %>%
  hc_add_theme(hc_theme_economist()) %>%
  hc_title(text="Top Contributors (Authors)")

## 4.6. Most Language Used ####################################################
scidata %>% 
  group_by(Language.of.Original.Document) %>%
  summarize(Total=n()) %>%
  plot_ly(labels=~Language.of.Original.Document,values=~Total,type="pie") %>%
  layout(title="Language Used ")

# 5. Text Analysis ############################################################

## 5.1. Index and Author Keywords Analysis ####################################

### 5.1.1. Frequent Keywords ##################################################
fq_auth <-colSums(dtmmauth)
fq_auth <-sort(fq_auth,decreasing = TRUE)
words_auth <-names(fq_auth)

fq_ind <-colSums(dtmmind)
fq_ind <-sort(fq_ind,decreasing = TRUE)
words_ind <-names(fq_ind)

### 5.1.2. Word Cloud Plotting ################################################
wordcloud(words_auth[1:200],
          fq_auth[1:200],random.order = FALSE,
          colors=brewer.pal(8,"Dark2"))

wordcloud(words_ind[1:200],fq_ind[1:200],
          random.order = FALSE,
          colors=brewer.pal(8,"Dark2"))

## 5.2. Article Titles ########################################################

### 5.2.1. LDA Model Testing ##################################################
k_list <- seq(5, 50, by = 5)
model_dirtit <- paste0("models_", digest::digest(vocabularytit, algo = "sha1"))
if (!dir.exists(model_dirtit)) dir.create(model_dirtit)

model_list_title <- TmParallelApply(X = k_list, FUN = function(k){
  filename = file.path(model_dirtit, paste0(k, "_topics.rda"))
  if (!file.exists(filename)) {
    mtit <- FitLdaModel(dtm = dtmtit, k = k, iterations = 500)
    mtit$k <- k
    mtit$coherence <- CalcProbCoherence(phi = mtit$phi, dtm = dtmtit, M = 5)
    save(mtit, file = filename)
  } else {
    load(filename)
  }
    mtit
}, export=c("dtmtit", "model_dirtit")) # export only needed for Windows machines

### 5.2.2. Coherence Matrix for LDA Models ####################################
coherence_mat_title <- data.frame(k = sapply(model_list_title, function(x) nrow(x$phi)), 
                                  coherence = sapply(model_list_title, function(x) mean(x$coherence)), 
                                  stringsAsFactors = FALSE)

plot(coherence_mat_title, type = "o")

### 5.2.3. Finding the Best K for LDA Model ###################################
model_title <- model_list_title[ which.max(coherence_mat_title$coherence) ][[ 1 ]]

names(model_title) # phi is P(words | topics), theta is P(topics | documents)

### 5.2.4. Top 10 Terms of the Model ##########################################
model_title$top_terms <- GetTopTerms(phi = model_title$phi, M = 10)
model_title$top_terms

### 5.2.5. Topics Analysis from Article Title #################################

#### 5.2.5.1. Topic Labels Using N-Gram from DTM ##############################
#Before that, we need to give a hard in/out assignment of topics in the documents
model_title$assignments <- model_title$theta
model_title$assignments[ model_title$assignments < 0.05 ] <- 0
model_title$assignments <- model_title$assignments / rowSums(model_title$assignments)
model_title$assignments[ is.na(model_title$assignments) ] <- 0

#Now that we have assignment to be assign, we can get topics from the DTM
model_title$labels <- LabelTopics(assignments = model_title$assignments, 
                            dtm = dtmtit,
                            M = 2)
head(model_title$labels)

#### 5.2.5.2 Topic Probabilistic Coherence ####################################
# Probabilistic Coherence: measures statistical support for a topic
model_title$coherence <- CalcProbCoherence(phi = model_title$phi, dtm = dtmtit, M = 5)
head(model_title$coherence)

#### 5.2.5.3. Number of Documents in which Each Topic Appears
model_title$num_docs <- colSums(model_title$assignments > 0)

### 5.2.6. Cluster Topics Together in a Dendrogram ############################
model_title$topic_linguistic_dist <- CalcHellingerDist(model_title$phi)
model_title$hclust <- hclust(as.dist(model_title$topic_linguistic_dist), "ward.D")
model_title$hclust$clustering <- cutree(model_title$hclust, k = 10)
model_title$hclust$labels <- paste(model_title$hclust$labels, model_title$labels[ , 1])

plot(model_title$hclust)
rect.hclust(model_title$hclust, k = length(unique(model_title$hclust$clustering)))

### 5.2.7. Summary Table for Article Titles ###################################
model_title$summary <- data.frame(topic = rownames(model_title$phi),
                                  cluster   = model_title$hclust$clustering,
                                  model_title$labels,
                                  coherence = model_title$coherence,
                                  num_docs  = model_title$num_docs,
                                  top_terms = apply(model_title$top_terms, 2, function(x){
                                    paste(x, collapse = ", ")
                                    }),
                                  stringsAsFactors = FALSE)

View(model_title$summary[ order(model_title$hclust$clustering) , ])
