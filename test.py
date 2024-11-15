from ToneAnalyzer import ToneAnalyzer


input_file_path = "Andrew.m4a"  # Replace with your .m4a or .wav file
converted_audio_path = ToneAnalyzer.convert_to_mono(input_file_path)
sentence = ToneAnalyzer.transcribe_audio(converted_audio_path)
print("Transcription:", sentence)
tone_analyzer = ToneAnalyzer(sentence)
sentiment_scores = tone_analyzer.analyze_tone()
stage = sentiment_scores['neg']
print(stage)
stage = (stage*3)+1

print(stage)

