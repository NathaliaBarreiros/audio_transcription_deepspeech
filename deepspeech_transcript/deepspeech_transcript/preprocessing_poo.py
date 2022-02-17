import contextlib
import wave
import collections
from deepspeech_transcript.preprocessing2 import read_wave
import webrtcvad
from typing import List, Dict, Tuple
from deepspeech import Model
from timeit import default_timer as timer
import glob
import os
import numpy as np
import pandas as pd

"""
class Wave

"""


class Wave:
    def __init__(self, path: str):
        self.path = path
        self.pcm_data = self.read_wave(path)["pcm_data"]
        self.sample_rate = self.read_wave(path)["sample_rate"]
        self.duration = self.read_wave(path)["duration"]

    def read_wave(self, path) -> Dict:
        with contextlib.closing(wave.open(path, "rb")) as wf:
            num_channels: int = wf.getnchannels()
            assert num_channels == 1
            sample_width: int = wf.getsampwidth()
            assert sample_width == 2
            sample_rate: int = wf.getframerate()
            assert sample_rate in (8000, 16000, 32000)
            frames: int = wf.getnframes()
            pcm_data: bytes = wf.readframes(frames)
            duration: float = frames / sample_rate

            return {"pcm_data": pcm_data, "sample_rate": sample_rate, "duration": duration}


def main():
    data = "../audios/audio1.wav"
    wave_instance = Wave(data)
    print(wave_instance.path)
    print(wave_instance.sample_rate)
    print(wave_instance.duration)


if __name__ == "__main__":
    main()
