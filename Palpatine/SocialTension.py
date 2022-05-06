import pandas as pd
import SentimentIntensityAnalyzer

class SocialTension:
    """"
    Class to assign 1, 0 or -1 value in context of sentiment. To get output, you need to invoke constructor
    SocialTension(text_array) in __main__ function and then
    """

    def __init__(self, text_array):
        self.text_array = pd.DataFrame(text_array, columns=['Text_array'])['Text_array']

    def sentiment_scores(self, sentence):
        """
        Function returns social tension of single text
        :return: integer
        """
        # Create a SentimentIntensityAnalyzer object.
        sid_obj = SentimentIntensityAnalyzer.SentimentIntensityAnalyzer()

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