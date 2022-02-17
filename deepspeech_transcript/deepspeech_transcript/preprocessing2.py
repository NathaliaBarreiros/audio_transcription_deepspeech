# %%
import contextlib
import wave
import collections
import webrtcvad
from typing import List, Dict
from deepspeech import Model
from timeit import default_timer as timer
import glob
import os
import numpy as np
import pandas as pd

# %%


class read_wave:
    def __init__(self, path: str):
        self.path = path
        print(self.path)

    def return_pcm(self):
        with contextlib.closing(wave.open(self.path, "rb")) as wf:
            self.num_channels: int = wf.getnchannels()
            assert self.num_channels == 1
            self.sample_width: int = wf.getsampwidth()
            assert self.sample_width == 2
            self.sample_rate: int = wf.getframerate()
            assert self.sample_rate in (8000, 16000, 32000)
            self.frames: int = wf.getnframes()
            self.pcm_data: bytes = wf.readframes(self.frames)
            self.duration: float = self.frames / self.sample_rate

            return self.pcm_data, self.sample_rate, self.duration

# %%


class Frame(object):
    def __init__(self, bytes: bytes, timestamp: float, duration: float):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


# %%
class frame_generator:
    def __init__(self, frame_duration_ms: int, audio: bytes, sample_rate: int):
        self.frame_duration_ms = frame_duration_ms
        self.audio = audio
        self.sample_rate = sample_rate

    def generate_frames(self):
        n = int(self.sample_rate * (self.frame_duration_ms / 1000.0) * 2)

        offset: int = 0
        timestamp: float = 0.0
        duration: float = (float(n) / self.sample_rate) / 2.0
        while offset + n < len(self.audio):
            yield Frame(self.audio[offset: offset + n], timestamp, duration)
            timestamp += duration
            offset += n

# %%


class vad_collector:
    def __init__(self, sample_rate: int, frame_duration_ms: int, padding_duration_ms: int, vad: webrtcvad.Vad, frames: List[Frame]):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.padding_duration_ms = padding_duration_ms
        self.vad = vad
        self.frames = frames

    def collector(self):
        num_padding_frames: int = int(
            self.padding_duration_ms / self.frame_duration_ms)
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered: bool = False
        voiced_frames = []

        for frame in self.frames:
            is_speech = self.vad.is_speech(frame.bytes, self.sample_rate)
            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        voiced_frames.append(f)
                    ring_buffer.clear()

            else:
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len(
                    [f for f, speech in ring_buffer if not speech])
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    triggered = False
                    yield b"".join([f.bytes for f in voiced_frames])
                    ring_buffer.clear()
                    voiced_frames = []

        if voiced_frames:
            yield b"".join([f.bytes for f in voiced_frames])


# %%
class vad_segment_generator:
    def __init__(self, wave_file: str, aggressiveness: int):
        self.wave_file = wave_file
        self.aggressiveness = aggressiveness

    def vad_generation(self):
        read_instance = read_wave(self.wave_file)
        self.audio, self.sample_rate, self.audio_length = read_instance.return_pcm()
        assert self.sample_rate == 16000, "Only 16000Hz input WAV files are supported for now!"
        vad = webrtcvad.Vad(int(self.aggressiveness))
        f_generator = frame_generator(30, self.audio, self.sample_rate)
        frames = f_generator.generate_frames()
        frames = list(frames)
        v_collector = vad_collector(self.sample_rate, 30, 300, vad, frames)
        self.segments = v_collector.collector()

        return self.segments, self.sample_rate, self.audio_length


# %%
class load_model:
    def __init__(self, models: str, scorer: str):
        self.models = models
        self.scorer = scorer

    def load(self):
        model_load_start = timer()
        ds = Model(self.models)
        model_load_end = timer() - model_load_start
        # print("Loaded model in %0.3fs." % (model_load_end))

        scorer_load_start = timer()
        ds.enableExternalScorer(self.scorer)
        scorer_load_end = timer() - scorer_load_start
        # print("Loaded external scorer in %0.3fs." % (scorer_load_end))

        return [ds, model_load_end, scorer_load_end]


# %%
class resolve_models:
    def __init__(self, dir_name: str):
        self.dir_name = dir_name

    def resolve(self):
        pb: str = glob.glob(self.dir_name + "/*.pbmm")[0]
        # print("Found Model: %s" % pb)

        scorer: str = glob.glob(self.dir_name + "/*.scorer")[0]
        # print("Found scorer: %s" % scorer)

        return pb, scorer


# %%
class stt_class:
    def __init__(self, ds: Model, audio: np.ndarray, fs: int):
        self.ds = ds
        self.audio = audio
        self.fs = fs
        print("STT CLASS TYPES: ")
        print(type(self.ds))
        print(type(self.audio))
        print(type(self.fs))

    def stt_func(self):
        self.inference_time: float = 0.0
        # Run Deepspeech
        # print("Running inference...")
        inference_start = timer()
        self.output: str = self.ds.stt(self.audio)
        inference_end = timer() - inference_start
        self.inference_time += inference_end
        # print(
        #     "Inference took %0.3fs for %0.3fs audio file." % (inference_end, audio_length)
        # )

        return [self.output, self.inference_time]


# %%
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

# %%
