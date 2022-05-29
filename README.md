AGH University of Science and Technology in Cracow
MSc Data Science Project
Data Mining Course

Contributors:
Marcin Świątkowski
Patryk Krukowski
Miłosz Włoch

# Twitter Sentiment Analysis in Polish 
## Introduction
The purpose of our project is to analyze tweets from Twitter with regards to a sentimental analysis. We collect tweets linked to a social tension in Poland caused by the war in the Ukraine. We want to do that by assigning a value one for tweets indicating positive emotions, a value zero in the case of lack of emotions, and a value minus one for tweets indicating negative emotions. This goal can be achieved by using a modification of the Vader algorithm which was created by Hutto, C.J. & Gilbert, E.E. (2014). The key concept of the modification of the Vader algorithm was to provide the functionality of the original Vader algorithm, but in Polish language.

## Contents:
* RT_ukraine_graphs.ipynb - Jupyter notebook to social networks analysis,
* SentiText.py - identifying sentiment-relevant string-level properties of input text,
SentimentIntensityAnalyzer.py - giving a sentiment intensity score to sentences,
VaderConstants.py - a class to keep the Vader lists and constants,
main.py - Python file with a drive function,
SocialTension.py - a class to assign 1, 0 or -1 value in context of sentiment,
Time_series_analysis.ipynb - Jupyter notebook to time series analysis,
cleaned_data.csv - data cleaning for tweets (regexs),
create_clusters.ipynb - location clusters of tweets,
create_word_cloud.ipynb - creating word cloud of given hashtags,
data_cleaning.ipynb - Jupyter notebook to filter the dataset,
get_tweets.ipynb - getting tweets with usage of Twitter API,
lemmatize_tweets.ipynb - used for tweets lemmatizing that is needed to get better output in Vader algorithm,
lemmatized_data.csv - lemmatized tweets’ text
output1.csv - all the gathered tweet data
pl.csv - coordinates of polish cities,
translate_words.ipynb - Jupyter notebook to translate tweets,
tweets_sentiments.csv - consists of tweets’ content and their sentiments,
ukraine-flag-national-europe-emblem-map-icon-illustration - a photo for word cloud
vader_lexicon_translated.txt - dictionary with polish words and emoticons with associated weights, 
odm.txt (https://sjp.pl/sl/odmiany/) - includes every form of polish words
#Data preprocessing
Downloading tweets:
We downloaded tweets using Twitter API. We choose these categories:  
tweet_id
used for ensuring that our database contains only unique tweets (or retweets)
text
main and the most important part of tweet
used for sentiment classification
for retweets: ‘text’ contains information about real author (source) of retweet
retweet_count
number of retweets of given tweet
created_at
date of creation of tweet or retweet of retweet
author
author of the tweet. It is important to remember that author means the user on which feed given tweet appears. It is not necessarily real author of tweet but it can be just the user who retweeted given tweet
author and source are used for creating network graphs
location
location of the user
is_verified
information whether user is verified or not

After we collected all the required data we needed to translate tweets’ text into polish language (translate_words.ipynb). Our algorithm uses odm.txt to lemmatize tweets (lemmatize_tweets.ipynb). The Odm.txt file consists of plenty of different Polish words with all of their forms. As we processed our data, then we could use the modified Vader algorithm to obtain tweets sentiments: –1 for negative sentiment, 0 for neutral one and 1 for positive one. The output of the algorithm was saved into the tweets_sentiments.csv file. 

#Executed Vader algorithm modifications
To achieve more information about how the Vader algorithm works please be sure to visit the following website: https://www.nltk.org/_modules/nltk/sentiment/vader.html
It was required to prepare this list of changes:
2.1. VaderConstants class:
We have decided to get rid of English idioms and every instruction connected to them.
Translating english words from the NEGATE set and the BOOSTER_DICT dictionary into polish language and appending some new words that can easily indicate specific emotions of the society.
2.2. SentiText class:
		-  To avoid an exception named AttributeError we have decided to add try - except block. The problem was related to converting float type to string type.
	2.3. SentimentIntensityAnalyzer class:
We had to match corresponding Polish words to English words, for example “but” -> “ale” etc.
	2.4. SocialTension class - that is the only one class which was added by us to the Vader package. The main target of this class was to measure tension of given tweets as an array-like object. The constructor can be invoked by passing array-like objects as an argument. In the result of this operation we have access to an attribute called text_array which is a DataFrame type. We have defined following methods:
sentiment_scores - determines sentiment score for given string:
-  input: string,
- output: 1, 0, -1.
measure_tension - measures tension for class attribute text_array:
-  input: -,
- output: pandas DataFrame.
	2.5. Function main() - driver function.
#Time series analysis
The time series analysis is stored in the  “Time_series_analysis.ipynb” jupyter notebook. There is an analysis and the corresponding conclusions of the tweets time series. Used declared functions:
moving_sum(pd_series):
argument: Pandas DataSeries,
output: array-like object, the function returns array-like object with moving sum of pd_series
plot_time_series(time_series, text = 'tension', title = 'Tweets', x_axis = 'Days')
- arguments: 
time_series - Pandas DataFrame with two columns (values of time series and an index - time)
 
text - a string which is appended at the end of  word “Total “; a name of the OY axis; default is equal to ‘tension’

title - a title of the plot; default is equal to ‘Tweets’

x_axis - a name of the OX axis
- output: a plot from the Plotly library
Social networks analysis
The social network analysis is stored in “RT_ukraine_graphs.ipynb” jupyter notebook. It contains analysis of network graphs, network motifs and sentimental distribution of information. Used declared functions:
((networkx graph is the input of each of these functions))

add_edges_to_graph(graph)
adds edges to newly created graph
transforms given graph adding edges to it
pseudo_diameter(graph)
calculates pseudo diameter (pseudo because our graph is not connected)
returns pseudo diameter
summary(graph)
prints summary of given graph
plot_degrees(graph)
plots bar chart of graph degrees
plot_centrality(graph)
plots bar chart of graph centrality
get_adj_matrix(graph)
returns adjacency matrix of given graph
plot_triplets(graph, order=[*range(0,16,1)])
input ‘order’ is a list of plotted motifs. Default it is set to plots all seventeen.
calculates and plots network motifs (triplets) of given graph
count_degrees(graph)
calculates and returns degrees for nodes in given graph
plot_chains(graph)
plots chains of nodes from given graph (works only for undirected graph and with no parallel edges)

