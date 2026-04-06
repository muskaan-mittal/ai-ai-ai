# Requirements: pip install transformers torch torchaudio soundfile

import torch
from transformers import pipeline

def main():
    # Example audio input (publicly available sample for demonstration)
    audio_input = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
    
    # Step 1: Transcribe the podcast audio into text
    print("Step 1: Transcribing audio...")
    asr_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
    asr_output = asr_pipe(audio_input)
    # Extract text from ASR output dictionary
    transcribed_text = asr_output["text"]
    print(f"Transcribed Text: {transcribed_text}\n")
    
    # Step 2: Summarize the transcribed text
    print("Step 2: Summarizing text...")
    sum_pipe = pipeline("summarization", model="facebook/bart-large-cnn")
    # Summarization pipeline returns a list of dicts
    sum_output = sum_pipe(transcribed_text)
    summarized_text = sum_output[0]["summary_text"]
    print(f"Summarized Text: {summarized_text}\n")
    
    # Step 3: Classify the topic of the summarized text
    print("Step 3: Classifying topic...")
    cls_pipe = pipeline("text-classification", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    # Text classification pipeline returns a list of dicts
    cls_output = cls_pipe(summarized_text)
    classification_result = cls_output[0]
    print(f"Classification Result: {classification_result}\n")

if __name__ == "__main__":
    main()