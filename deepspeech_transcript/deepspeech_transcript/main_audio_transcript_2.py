# %% Libraries import
from operator import index
from deepspeech import Model
from timeit import default_timer as timer
import os
import numpy as np
import preprocessing2 as prp
import pandas as pd
from typing import List


def main():
    data = "../audios/audio1.wav"
    aggressive = 1

    model: str = "~/audio_transcription_deepspeech/deepspeech_transcript/models"
    dir_name = os.path.expanduser(model)
    # print("DIR NAME TYPE")
    # print(type(dir_name))

    vad_instance = vad_segment_generator(data, aggressive)
    segments, sample_rate, audio_length = vad_instance.vad_generation()

    print("VAD SEGMENT GENERATOR WORKS!!!!!!")
    print(segments)
    print(sample_rate)
    print(audio_length)

    resolve_instance = resolve_models(dir_name)
    output_graph, scorer = resolve_instance.resolve()

    print("RESOLVE CLASS WORKS!!!!")
    print(output_graph)
    print(scorer)
    print(type(output_graph))
    print(type(scorer))

    load_instance = load_model(output_graph, scorer)
    model_retval = load_instance.load()

    print("LOAD CLASS WORKS!!!")
    print(model_retval)
    print(type(model_retval))

    for j, segment in enumerate(segments):
        audio = np.frombuffer(segment, dtype=np.int16)
        stt_instance = stt_class(model_retval[0], audio, sample_rate)

        output: List[str, float] = stt_instance.stt_func()
        transcript: str = output[0]
    print("STT CLASS WORKS!!!")
    print(transcript)
    print(type(transcript))
    print(type(output[1]))

    # transcriptions.append(transcript)

    # read_instance = read_wave(data)
    # audio, sample_rate, audio_length = read_instance.return_pcm()
    # # print(audio)
    # print("READ_WAVE CLASS WORKS!!")
    # print(sample_rate)
    # print(audio_length)
    # f_generator = frame_generator(30, audio, sample_rate)
    # frames = f_generator.generate_frames()
    # frames = list(frames)
    # print("FRAMES: ")
    # print(frames)
    # print("FRAMES_GENERATOR CLASS WORKS!!")
    # vad = webrtcvad.Vad(int(1))
    # v_collector = vad_collector(sample_rate, 30, 300, vad, frames)
    # print("SEGMENTS")
    # segments = v_collector.collector()
    # print(segments)
    # print(type(segments))
    # return data, sample_rate, audio_length


if __name__ == "__main__":
    main()
