# %% Libraries import
from deepspeech_transcript.preprocessing_poo import Preprocessing
from deepspeech_transcript.preprocessing_poo import DeepSpeechModel
import numpy as np
import pandas as pd
from typing import List
from operator import index
from deepspeech import Model
import os


# %%
def main():

    model: str = "~/audio_transcription_deepspeech/deepspeech_transcript/models"
    dir_name = os.path.expanduser(model)
    ds_instance = DeepSpeechModel(dir_name)
    output_graph, scorer = ds_instance.resolve_models_paths()
    model_retval: List[str] = ds_instance.load_models()
    data = pd.read_csv("../csv_docs/paths.csv")
    paths: List[str] = list(data.audio_path)

    transcriptions: List[str] = []
    new_data: List[str] = []
    final_data: List[List] = []
    header: List[str] = ["audio_path", "transcriptions"]
    aggresive = 1

    print("Running inference...")

    for i in range(len(data.audio_path)):
        audio = data.audio_path[i]
        wave_file = audio
        wave_instance = Preprocessing(wave_file, 30, 300, aggresive)
        segments, sample_rate, audio_length, vad = wave_instance.vad_segment_generator(
            wave_file, aggresive)
        for j, segment in enumerate(segments):
            audio = np.frombuffer(segment, dtype=np.int16)
            output = ds_instance.transcript_audio_segments(
                model_retval[0], audio)
            transcript = output[0]
        transcriptions.append(transcript)
        new_data = [paths[i], transcriptions[i]]
        final_data.append(new_data)

    new_csv = pd.DataFrame(final_data, columns=header)
    new_csv.to_csv("../csv_docs/transcripted_audios.csv", index=False)

    print("Audio transcriptions are done, go fetch the transcripted_audios.csv file!")


# %%
if __name__ == "__main__":
    main()
