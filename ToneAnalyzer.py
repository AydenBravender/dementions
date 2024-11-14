from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class ToneAnalyzer:
    def __init__(self, sentence):
        self.sentence = sentence

    def analyze_tone(self):
        # Initialize VADER sentiment analyzer
        analyzer = SentimentIntensityAnalyzer()
        # Analyze sentiment of the stored sentence
        sentiment_scores = analyzer.polarity_scores(self.sentence)
        return sentiment_scores
