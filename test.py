from ToneAnalyzer import ToneAnalyzer

sentence = "your gay as hell"
tone_analyzer = ToneAnalyzer(sentence)
sentiment_scores = tone_analyzer.analyze_tone()
print(sentiment_scores)