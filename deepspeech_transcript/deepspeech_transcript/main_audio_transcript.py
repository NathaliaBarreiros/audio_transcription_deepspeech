# %% Libraries import
from operator import index
from deepspeech import Model
from timeit import default_timer as timer
import os
import numpy as np
import preprocessing as pr
import pandas as pd
from typing import List


# %% Main function
def main():
    model: str = "~/audio_transcription_deepspeech/deepspeech_transcript/models"
    dir_name = os.path.expanduser(model)

    # input("What level of non-voice filtering would you like? (0-3)")
    aggressive = 1

    # Resolve all the paths of model files
    output_graph, scorer = pr.resolve_models(dir_name)

    # Load output_graph, alphabet and scorer
    model_retval: List[str] = pr.load_model(output_graph, scorer)

    # For jupyter & python file
    data = pd.read_csv("../csv_docs/paths.csv")

    # For debugger
    # data = pd.read_csv("./deepspeech_transcript/csv_docs/paths.csv")

    paths: List[str] = list(data.audio_path)

    transcriptions: List[str] = []
    new_data: List[str] = []
    final_data: List[List] = []

    header: List[str] = ["audio_path", "transcriptions"]

    print("Running inference...")
    for i in range(len(data.audio_path)):
        audio = data.audio_path[i]
        # print(audio)

        wave_file = audio
        segments, sample_rate, audio_length = pr.vad_segment_generator(
            wave_file, aggressive
        )
        for j, segment in enumerate(segments):
            audio = np.frombuffer(segment, dtype=np.int16)
            output = pr.stt(model_retval[0], audio, sample_rate)
            transcript = output[0]
        transcriptions.append(transcript)
        new_data = [paths[i], transcriptions[i]]
        final_data.append(new_data)

    new_csv = pd.DataFrame(final_data, columns=header)
    new_csv.to_csv("../csv_docs/transcripted_audios3.csv", index=False)

    # For debugger
    # new_csv.to_csv(
    #     "./deepspeech_transcript/csv_docs/transcripted_audios_p2.csv", index=False)

    print("Audio transcriptions are done, go fetch the transcripted_audios.csv file!")


# %% Entry point
if __name__ == "__main__":
    main()

# %%
