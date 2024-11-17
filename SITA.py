#speech impairment and tone analysis

import whisper
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import wave
import json
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import os
import google.generativeai as genai
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import contextlib

#AudioSeg path bin config (FOR VOSK)
AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"

#Vosk model path config
vosk_model_path = r"C:\Vosk models\vosk-model-small-en-us-0.15"

#API Key Config
API_KEY = "AIzaSyAG58lnAZeH68lavqYJ0g3sfV6gI4kvIkg"
genai.configure(api_key=API_KEY)
#google ai model config
model = genai.GenerativeModel("gemini-1.5-flash")

class Sita:
    def __init__(self, filepath):
        self.audio = filepath
        converted_audio = self.convert_to_mono(self.audio)
        self.sentence = self.whisper_transcribe(converted_audio) #CAN REPLACE WITH VOSK TRANSCRIBE
        self.gemini_score = self.gemini_prompt(self.sentence)
        self.sentiment_scores = self.analyze_tone(self.sentence)
        self.wps = self.find_wps(converted_audio, self.sentence) #not a score, score_calc function takes care of that, just sloppy coding on my end, and too lazy - Lucas
        self.score_pair = self.score_calc(self.wps, self.gemini_score, self.sentiment_scores)
        self.outlier_sentences = self.tokenization(self.sentence)


    def analyze_tone(self, sentence):
        # Initialize VADER sentiment analyzer
        analyzer = SentimentIntensityAnalyzer()
        # Analyze sentiment of the stored sentence
        sentiment_scores = analyzer.polarity_scores(sentence)
        score = sentiment_scores['neg']*100
        return score
    
    def gemini_prompt(self, text):
        prompt = "\n Does the text above suggest that the speaker has dementia, on a scale of 0-100? Please respond with only an integer, 0 being no dementia at all, and 100 being severe dementia"
        response = model.generate_content(text + prompt)
        output = int(response.text.split()[0])
        return output
    
    def whisper_transcribe(self, file_path):
        #Load whisper model
        model = whisper.load_model("base")
        #transribe using wav file
        result = model.transcribe(file_path)
        return result["text"]

    def tokenization(self, text):
        #load model
        nlp = spacy.load("en_core_web_sm")
        #load sentencizer
        sentencizer = nlp.add_pipe("sentencizer")
        #turn into a nlp workable object
        doc = nlp(text)
        #threshold for context score, anything below threshold will be flagged as irrelevant
        threshold = 0.5

        #tokenize
        tokens = [token.text for token in doc]
        token_vectors = [token.vector for token in doc]

        #embeddings
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        sentences = [str(sentence) for sentence in doc.sents] #doc.sents is an interator, we are creating a list of sentence objects at once so we do not have to reiterate doc.sents
        embeddings = np.array(model.encode(sentences))

        #average embeddings
        topic_vector = np.mean(embeddings, axis=0).reshape(1, -1)

        similarity_scores = cosine_similarity(embeddings, topic_vector).flatten()

        irrelevant_sentences = [
            (sentence, score)
            for sentence, score in zip(sentences, similarity_scores)
            if score < threshold
        ]
        #return an array of irrelevant sentecnes
        return irrelevant_sentences

    def vosk_transcribe(self, file_path, model_path=vosk_model_path):
        # Load the Vosk model
        model = Model(model_path)
        
        # Open the audio file
        wf = wave.open(file_path, "rb")
        
        # Ensure compatibility of audio file format
        if wf.getframerate() not in [8000, 16000, 32000, 44100, 48000]:
            raise ValueError("Audio file must have a sample rate of 8000, 16000, 32000, 44100, or 48000 Hz.")
        if wf.getnchannels() != 1:
            raise ValueError("Audio file must be mono channel.")
        
        # Initialize the recognizer with the sample rate
        recognizer = KaldiRecognizer(model, wf.getframerate())
        recognizer.SetGrammar('["uhhh", "umm"]')  

        # Read the audio file and transcribe
        transcription = ""
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                transcription += json.loads(result)["text"] + " "
        
        # Get final text from recognizer
        transcription += json.loads(recognizer.FinalResult())["text"]
        return transcription

    #input .wav file and transcript
    def find_wps(self, converted_file, text):
        with contextlib.closing(wave.open(converted_file,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)

        wps = len(text.split()) / duration

        return wps

    @staticmethod
    def convert_to_mono(input_file, output_file="converted_mono_audio.wav", target_rate=16000):
        if not isinstance(input_file, str):
            raise ValueError(f"Expected a file path (str), got {type(input_file)}")
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"File not found: {input_file}")
        #Load the audio file with pydub (supports m4a)
        audio = AudioSegment.from_file(input_file)
        # Convert to mono and set the sample rate
        audio = audio.set_channels(1).set_frame_rate(target_rate)
        # Export as wav
        audio.export(output_file, format="wav")
        print(f"Converted audio saved as {output_file} with sample rate {target_rate} Hz and mono channel.")
        return output_file
    
    #tone should be result of class.analyze_tone()
    def score_calc(self, wps, gemini, tone):
        wps = (wps/1.9) * 100
        if wps > 100:
            wps = 100
        speech_cohesiveness = (gemini * 0.7) + (wps * 0.3)
        return speech_cohesiveness, tone
    
toneanal = Sita("ResponseAudios/MergedAudio.wav")
print(toneanal.score_pair)