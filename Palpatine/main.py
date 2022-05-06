# Natural Language Toolkit: vader
#
# Copyright (C) 2001-2022 NLTK Project
# Author: C.J. Hutto <Clayton.Hutto@gtri.gatech.edu>
#         Ewan Klein <ewan@inf.ed.ac.uk> (modifications)
#         Pierpaolo Pantone <24alsecondo@gmail.com> (modifications)
#         George Berry <geb97@cornell.edu> (modifications)
#         Malavika Suresh <malavika.suresh0794@gmail.com> (modifications)
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
#
# Modifications to the original VADER code have been made in order to
# integrate it into NLTK. These have involved changes to
# ensure Python 3 compatibility, and refactoring to achieve greater modularity.

"""
If you use the VADER sentiment analysis tools, please cite:

Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for
Sentiment Analysis of Social Media Text. Eighth International Conference on
Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
"""

import pandas as pd
import SentimentIntensityAnalyzer
import SocialTension

if __name__ == '__main__':
    tweets = []
    df = pd.read_csv('lemmatized_data.csv', encoding='utf-8')
    for index, text in df.iterrows():
        tweets.append(df['text'][index])
    analyzer = SentimentIntensityAnalyzer.SentimentIntensityAnalyzer()
    tension = SocialTension.SocialTension(tweets)
    result = tension.measure_tension()
    result.to_csv('tweets_sentiments.csv', index=False)
