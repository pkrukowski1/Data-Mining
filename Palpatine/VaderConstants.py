import re
import string
import math

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
