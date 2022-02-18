# %% Libraries import
from operator import index
from deepspeech import Model
import os
import numpy as np
import pandas as pd
from typing import List
import deepspeech_transcript.functional_programming_paradigm.preprocessing as pr


# %% Main function
def main():
    model: str = "~/audio_transcription_deepspeech/deepspeech_transcript/models"
    dir_name = os.path.expanduser(model)

    # Resolves all the paths of model files
    output_graph, scorer = pr.resolve_models(dir_name)

    # Loads output_graph and scorer
    model_retval: List[str] = pr.load_model(output_graph, scorer)

    data = pd.read_csv("../csv_docs/paths.csv")
    aggressive = 1
    paths: List[str] = list(data.audio_path)
    transcriptions: List[str] = []
    new_data: List[str] = []
    final_data: List[List] = []
    header: List[str] = ["audio_path", "transcriptions"]

    print("Running inference...")

    for i in range(len(data.audio_path)):
        audio = data.audio_path[i]
        wave_file = audio
        # Instance of Preprocessing class to generate vad segments of pcm audio
        segments, sample_rate, audio_length = pr.vad_segment_generator(
            wave_file, aggressive
        )
        for j, segment in enumerate(segments):
            audio = np.frombuffer(segment, dtype=np.int16)
            # Vad audio segments transcription
            output = pr.stt(model_retval[0], audio)
            transcript = output[0]
        transcriptions.append(transcript)
        new_data = [paths[i], transcriptions[i]]
        final_data.append(new_data)
    # Audio paths and audio's transcriptions into new CSV file
    new_csv = pd.DataFrame(final_data, columns=header)
    new_csv.to_csv("../csv_docs/transcripted_audios.csv", index=False)

    print("Audio transcriptions are done, go fetch the transcripted_audios.csv file!")


# %% Entry point
if __name__ == "__main__":
    main()
