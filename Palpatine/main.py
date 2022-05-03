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

import math
from itertools import product
import pandas as pd
import re
import string
import morfeusz2
import nltk
import nltk.data
from nltk.util import pairwise


class VaderConstants:
    """
    A class to keep the Vader lists and constants.
    """

    ##Constants##
    # (empirically derived mean sentiment intensity rating increase for booster words)
    B_INCR = 0.293
    B_DECR = -0.293

    # (empirically derived mean sentiment intensity rating increase for using
    # ALLCAPs to emphasize a word)
    C_INCR = 0.733

    N_SCALAR = -0.74

    NEGATE = {
        "nie",
        "rzadki",
        "bez",
    }

    # booster/dampener 'intensifiers' or 'degree adverbs'
    # https://en.wiktionary.org/wiki/Category:English_degree_adverbs

    BOOSTER_DICT = {
        "absolutny": B_INCR,
        "niesamowity": B_INCR,
        "świetny": B_INCR,
        "okropny": B_INCR,
        "całkowity": B_INCR,
        "znaczny": B_INCR,
        "zdecydować": B_INCR,
        "głęboki": B_INCR,
        "przekląć": B_INCR,
        "cholerny": B_INCR,
        "ogromny": B_INCR,
        "całkiem": B_INCR,
        "szczegółowy": B_INCR,
        "wyjątkowy": B_INCR,
        "bajeczny": B_INCR,
        "trzepnąć": B_INCR,
        "zwariowany": B_INCR,
        "pieprzyć": B_INCR,
        "pełny": B_INCR,
        "zjebać": B_INCR,
        "wspaniały": B_INCR,
        "wysoki": B_INCR,
        "duży": B_INCR,
        "intensywny": B_INCR,
        "więcej": B_INCR,
        "najwięcej": B_INCR,
        "wyłączny": B_INCR,
        "cichy": B_INCR,
        "naprawdę": B_INCR,
        "znaczący": B_INCR,
        "dosycić": B_INCR,
        "zasadniczy": B_INCR,
        "ogólny": B_INCR,
        "wyobrażalny": B_INCR,
        "bardzo": B_INCR,
        "prawo": B_DECR,
        "ledwo": B_DECR,
        "wystarczać": B_DECR,
        "cząstkowy": B_DECR,
        "poniekąd": B_DECR,
        "mniej": B_DECR,
        "marginalny": B_DECR,
        "okazjonalny": B_DECR,
        "częściowy": B_DECR,
        "troszkę": B_DECR,
        "nieco": B_DECR,
        "odrobinę": B_DECR,
        "trochę": B_DECR,
        "kurwa": B_INCR,
        "pierdolić" : B_INCR,
        "wyjebać" : B_INCR,
        "zajebać" : B_INCR,
        "podjebać" : B_INCR,
        "opierdolić" : B_INCR,
        "po chuja" : B_INCR,
        "chuj" : B_INCR,
        "chujowy" : B_INCR,
        "spierdalać" : B_INCR,
        "spierdolić" : B_INCR,
        "chujec" : B_INCR,
    }

    # for removing punctuation
    REGEX_REMOVE_PUNCTUATION = re.compile(f"[{re.escape(string.punctuation)}]")

    PUNC_LIST = [
        ".",
        "!",
        "?",
        ",",
        ";",
        ":",
        "-",
        "'",
        '"',
        "!!",
        "!!!",
        "??",
        "???",
        "?!?",
        "!?!",
        "?!?!",
        "!?!?",
    ]

    def __init__(self):
        pass

    def negated(self, input_words, include_nt=True):
        """
        Determine if input contains negation words
        """
        neg_words = self.NEGATE
        if any(word.lower() in neg_words for word in input_words):
            return True
        return False


    def normalize(self, score, alpha=15):
        """
        Normalize the score to be between -1 and 1 using an alpha that
        approximates the max expected value
        """
        norm_score = score / math.sqrt((score * score) + alpha)
        return norm_score


    def scalar_inc_dec(self, word, valence, is_cap_diff):
        """
        Check if the preceding words increase, decrease, or negate/nullify the
        valence
        """
        scalar = 0.0
        word_lower = word.lower()
        if word_lower in self.BOOSTER_DICT:
            scalar = self.BOOSTER_DICT[word_lower]
            if valence < 0:
                scalar *= -1
            # check if booster/dampener word is in ALLCAPS (while others aren't)
            if word.isupper() and is_cap_diff:
                if valence > 0:
                    scalar += self.C_INCR
                else:
                    scalar -= self.C_INCR
        return scalar



class SentiText:
    """
    Identify sentiment-relevant string-level properties of input text.
    """

    def __init__(self, text, punc_list, regex_remove_punctuation):
        if not isinstance(text, str):
            text = str(text.encode("utf-8"))
        self.text = text
        self.PUNC_LIST = punc_list
        self.REGEX_REMOVE_PUNCTUATION = regex_remove_punctuation
        self.words_and_emoticons = self._words_and_emoticons()
        # doesn't separate words from
        # adjacent punctuation (keeps emoticons & contractions)
        self.is_cap_diff = self.allcap_differential(self.words_and_emoticons)


    def _words_plus_punc(self):
        """
        Returns mapping of form:
        {
            'cat,': 'cat',
            ',cat': 'cat',
        }
        """
        no_punc_text = self.REGEX_REMOVE_PUNCTUATION.sub("", self.text)
        # removes punctuation (but loses emoticons & contractions)
        words_only = no_punc_text.split()
        # remove singletons
        words_only = {w for w in words_only if len(w) > 1}
        # the product gives ('cat', ',') and (',', 'cat')
        punc_before = {"".join(p): p[1] for p in product(self.PUNC_LIST, words_only)}
        punc_after = {"".join(p): p[0] for p in product(words_only, self.PUNC_LIST)}
        words_punc_dict = punc_before
        words_punc_dict.update(punc_after)
        return words_punc_dict

    def _words_and_emoticons(self):
        """
        Removes leading and trailing puncutation
        Leaves contractions and most emoticons
            Does not preserve punc-plus-letter emoticons (e.g. :D)
        """
        wes = self.text.split()
        words_punc_dict = self._words_plus_punc()
        wes = [we for we in wes if len(we) > 1]
        for i, we in enumerate(wes):
            if we in words_punc_dict:
                wes[i] = words_punc_dict[we]
        return wes

    def allcap_differential(self, words):
        """
        Check whether just some words in the input are ALL CAPS

        :param list words: The words to inspect
        :returns: `True` if some but not all items in `words` are ALL CAPS
        """
        is_different = False
        allcap_words = 0
        for word in words:
            if word.isupper():
                allcap_words += 1
        cap_differential = len(words) - allcap_words
        if 0 < cap_differential < len(words):
            is_different = True
        return is_different



class SentimentIntensityAnalyzer:
    """
    Give a sentiment intensity score to sentences.
    """

    def __init__(
        self,
        lexicon_file="vader_lexicon_translated.txt",
    ):
        self.lexicon_file = nltk.data.load(lexicon_file)
        self.lexicon = self.make_lex_dict()
        self.constants = VaderConstants()


    def make_lex_dict(self):
        """
        Convert lexicon file to a dictionary
        """
        lex_dict = {}
        for line in self.lexicon_file.split("\n"):
            if len(line) < 2:
                continue
            (word, measure) = line.strip().split("+")[0:2]
            lex_dict[word] = float(measure)
        return lex_dict


    def polarity_scores(self, text):
        """
        Return a float for sentiment strength based on the input text.
        Positive values are positive valence, negative value are negative
        valence.
        """
        # text, words_and_emoticons, is_cap_diff = self.preprocess(text)
        sentitext = SentiText(
            text, self.constants.PUNC_LIST, self.constants.REGEX_REMOVE_PUNCTUATION
        )
        sentiments = []
        words_and_emoticons = sentitext.words_and_emoticons
        for item in words_and_emoticons:
            valence = 0
            i = words_and_emoticons.index(item)
            if item.lower() in self.constants.BOOSTER_DICT:
                sentiments.append(valence)
                continue

            sentiments = self.sentiment_valence(valence, sentitext, item, i, sentiments)

        sentiments = self._but_check(words_and_emoticons, sentiments)

        return self.score_valence(sentiments, text)


    def sentiment_valence(self, valence, sentitext, item, i, sentiments):
        is_cap_diff = sentitext.is_cap_diff
        words_and_emoticons = sentitext.words_and_emoticons
        item_lowercase = item.lower()
        if item_lowercase in self.lexicon:
            # get the sentiment valence
            valence = self.lexicon[item_lowercase]

            # check if sentiment laden word is in ALL CAPS (while others aren't)
            if item.isupper() and is_cap_diff:
                if valence > 0:
                    valence += self.constants.C_INCR
                else:
                    valence -= self.constants.C_INCR

            for start_i in range(0, 3):
                if (
                    i > start_i
                    and words_and_emoticons[i - (start_i + 1)].lower()
                    not in self.lexicon
                ):
                    # dampen the scalar modifier of preceding words and emoticons
                    # (excluding the ones that immediately preceed the item) based
                    # on their distance from the current item.
                    s = self.constants.scalar_inc_dec(
                        words_and_emoticons[i - (start_i + 1)], valence, is_cap_diff
                    )
                    if start_i == 1 and s != 0:
                        s = s * 0.95
                    if start_i == 2 and s != 0:
                        s = s * 0.9
                    valence = valence + s
                    valence = self._never_check(
                        valence, words_and_emoticons, start_i, i
                    )

            valence = self._least_check(valence, words_and_emoticons, i)

        sentiments.append(valence)
        return sentiments


    def _least_check(self, valence, words_and_emoticons, i):
        # check for negation case using "least"
        if (
            i > 1
            and words_and_emoticons[i - 1].lower() not in self.lexicon
        ):
            if (
                i > 0
                and words_and_emoticons[i - 1].lower() not in self.lexicon
            ):
                valence = valence * self.constants.N_SCALAR
        return valence

    def _but_check(self, words_and_emoticons, sentiments):
        words_and_emoticons = [w_e.lower() for w_e in words_and_emoticons]
        but = {"ale"} & set(words_and_emoticons)
        if but:
            bi = words_and_emoticons.index(next(iter(but)))
            for sidx, sentiment in enumerate(sentiments):
                if sidx < bi:
                    sentiments[sidx] = sentiment * 0.5
                elif sidx > bi:
                    sentiments[sidx] = sentiment * 1.5
        return sentiments


    def _never_check(self, valence, words_and_emoticons, start_i, i):
        if start_i == 0:
            if self.constants.negated([words_and_emoticons[i - 1]]):
                valence = valence * self.constants.N_SCALAR
        if start_i == 1:
            if words_and_emoticons[i - 2] == "nigdy" and (
                words_and_emoticons[i - 1] == "więc"
                or words_and_emoticons[i - 1] == "to"
            ):
                valence = valence * 1.5
            elif self.constants.negated([words_and_emoticons[i - (start_i + 1)]]):
                valence = valence * self.constants.N_SCALAR
        if start_i == 2:
            if (
                words_and_emoticons[i - 3] == "nigdy"
                and (
                    words_and_emoticons[i - 2] == "więc"
                    or words_and_emoticons[i - 2] == "to"
                )
                or (
                    words_and_emoticons[i - 1] == "więc"
                    or words_and_emoticons[i - 1] == "to"
                )
            ):
                valence = valence * 1.25
            elif self.constants.negated([words_and_emoticons[i - (start_i + 1)]]):
                valence = valence * self.constants.N_SCALAR
        return valence

    def _punctuation_emphasis(self, sum_s, text):
        # add emphasis from exclamation points and question marks
        ep_amplifier = self._amplify_ep(text)
        qm_amplifier = self._amplify_qm(text)
        punct_emph_amplifier = ep_amplifier + qm_amplifier
        return punct_emph_amplifier

    def _amplify_ep(self, text):
        # check for added emphasis resulting from exclamation points (up to 4 of them)
        ep_count = text.count("!")
        if ep_count > 4:
            ep_count = 4
        # (empirically derived mean sentiment intensity rating increase for
        # exclamation points)
        ep_amplifier = ep_count * 0.292
        return ep_amplifier

    def _amplify_qm(self, text):
        # check for added emphasis resulting from question marks (2 or 3+)
        qm_count = text.count("?")
        qm_amplifier = 0
        if qm_count > 1:
            if qm_count <= 3:
                # (empirically derived mean sentiment intensity rating increase for
                # question marks)
                qm_amplifier = qm_count * 0.18
            else:
                qm_amplifier = 0.96
        return qm_amplifier

    def _sift_sentiment_scores(self, sentiments):
        # want separate positive versus negative sentiment scores
        pos_sum = 0.0
        neg_sum = 0.0
        neu_count = 0
        for sentiment_score in sentiments:
            if sentiment_score > 0:
                pos_sum += (
                    float(sentiment_score) + 1
                )  # compensates for neutral words that are counted as 1
            if sentiment_score < 0:
                neg_sum += (
                    float(sentiment_score) - 1
                )  # when used with math.fabs(), compensates for neutrals
            if sentiment_score == 0:
                neu_count += 1
        return pos_sum, neg_sum, neu_count

    def score_valence(self, sentiments, text):
        if sentiments:
            sum_s = float(sum(sentiments))
            # compute and add emphasis from punctuation in text
            punct_emph_amplifier = self._punctuation_emphasis(sum_s, text)
            if sum_s > 0:
                sum_s += punct_emph_amplifier
            elif sum_s < 0:
                sum_s -= punct_emph_amplifier

            compound = self.constants.normalize(sum_s)
            # discriminate between positive, negative and neutral sentiment scores
            pos_sum, neg_sum, neu_count = self._sift_sentiment_scores(sentiments)

            if pos_sum > math.fabs(neg_sum):
                pos_sum += punct_emph_amplifier
            elif pos_sum < math.fabs(neg_sum):
                neg_sum -= punct_emph_amplifier

            total = pos_sum + math.fabs(neg_sum) + neu_count
            pos = math.fabs(pos_sum / total)
            neg = math.fabs(neg_sum / total)
            neu = math.fabs(neu_count / total)

        else:
            compound = 0.0
            pos = 0.0
            neg = 0.0
            neu = 0.0

        sentiment_dict = {
            "neg": round(neg, 3),
            "neu": round(neu, 3),
            "pos": round(pos, 3),
            "compound": round(compound, 4),
        }

        return sentiment_dict


class SocialTension:
    """"
    Class to assign 1, 0 or -1 value in context of sentiment. To get output, you need to invoke constructor
    SocialTension(text_array) in __main__ function and then
    """

    def __init__(self, text_array):
        self.text_array = pd.DataFrame(text_array, columns=['Text_array'])['Text_array'].apply(lambda x : self.lemme_lemmatize(x))

    def remove_digits_from_words(self, text):
        numbers = [str(i) for i in range(0, 10)]
        letters = list(string.ascii_lowercase)
        for word in text.split():
            for num in numbers:
                for letter in letters:
                    if num in word and letter in word:
                        new_word = re.split('(\d+)', word)[0]
                        text = text.replace(word, new_word, 1)
        return text

    def no_puncs(self, tweet):
        tweet_done = tweet.translate(str.maketrans('', '', string.punctuation)).lower()
        return tweet_done

    def make_list(self, string):
        mylist = string.split()
        return mylist

    def remove_duplicates(self, text):
        new_text = []
        for word in text.split():
            if ":" in word:
                word_halves = word.split(':')
                word = word_halves[0]
                new_text.append(word)
            else:
                new_text.append(word)
        mylist = list(dict.fromkeys(new_text))
        mylist = ' '.join(mylist)
        return mylist

    def lemme_lemmatize(self, text):
        done_text = []  # miejsce na output
        text = text.lower()  # konwersja na małe znaki
        text = self.no_puncs(text)  # usunięcie interpunkcji
        text = self.remove_digits_from_words(text)  # usunięcie cyfr złączonych ze słowami

        morf = morfeusz2.Morfeusz()  # stworzenie obiektu morfeusza
        analysis = morf.analyse(text)  # stworzenie obiektu analizującego na bazie textu
        for i, j, interp in analysis:
            done_text.append(interp[1])  # dodanie lematu do outputu
        done_text = ' '.join(done_text)  # przerobienie outputu na stringa
        done_text = self.remove_duplicates(done_text)  # usunięcie zduplikowanych lematów
        #     done_text = make_list(done_text) # opcjonalnie, zwraca output jako listę
        return done_text

    def sentiment_scores(self, sentence):
        """
        Function returns social tension of single text
        :return: integer
        """
        # Create a SentimentIntensityAnalyzer object.
        sid_obj = SentimentIntensityAnalyzer()

        # polarity_scores method of SentimentIntensityAnalyzer
        # object gives a sentiment dictionary.
        # which contains pos, neg, neu, and compound scores.
        sentiment_dict = sid_obj.polarity_scores(sentence)


        # decide sentiment as positive, negative and neutral
        if sentiment_dict['compound'] >= 0.05:
            return 1

        elif sentiment_dict['compound'] <= - 0.05:
            return -1

        else:
            return 0

    def measure_tension(self):
        """
        Function returns pandas data frame with assigned social tension values represented by 1, 0 and -1
        :return: DataFrame
        """
        result = {'Text' : self.text_array, 'Tension' : [0 for _ in range(len(self.text_array))]}
        result = pd.DataFrame(result)
        result['Tension'] = result['Text'].apply(self.sentiment_scores)

        return result

if __name__ == '__main__':
    tweets = []
    df = pd.read_csv('output1.csv')
    for index, text in df.iterrows():
        tweets.append(df['text'][index])
    analyzer = SentimentIntensityAnalyzer()
    tension = SocialTension(tweets)
    result = tension.measure_tension()
    result.to_csv('tweets_sentiments.csv', index=False)
