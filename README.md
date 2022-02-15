# Transcription .wav audio files with DeepSpeech library

The objective of this project is to read a column of a CSV file that contains paths of audio files with a .wav extension, and then transcribe the audios with the help of Mozilla's DeepSpeech library. Finally, another CSV file containing a new column with the transcribed text must be returned.

## Audio specifications

---

Since last DeepSpeed version only supported 16kHz .wav files, audio files must fulfill this requirement. Also, the preproccessing work expects 1 channel English audio files.

## Scripts specifications

---

- preprocessing.py file: This script is based on vad_transcriber DeepSpeed exaple https://github.com/mozilla/DeepSpeech-examples/tree/r0.9/vad_transcriber. This is in charged of managing .wav files and converting those into segments of voiced PCM audio data that can be processed by DeepSpeech ML model. It also has functions for loading and resolve models.

- main_audio_transcript.py file: This script iterates over data of the initial CSV file to obtain the audio paths to be transcripted, processes and transcriptes them, using the preprocessing.py functions, and returns a new CSV file with the audio paths and the transcripts.

## Pre-trained English model files used

---

- curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm
- curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer
