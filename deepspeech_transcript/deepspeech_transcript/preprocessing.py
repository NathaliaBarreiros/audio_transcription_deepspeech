#%% Libraries import
import collections
import contextlib
import wave
import webrtcvad
import sys
from deepspeech import Model
from timeit import default_timer as timer
import glob
import os
import numpy as np
from typing import List

#%% Read .wav file function, returns (PCM audio data, sample rate, duration)
def read_wave(path):
    with contextlib.closing(wave.open(path, "rb")) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000)
        frames = wf.getnframes()
        pcm_data = wf.readframes(frames)
        duration = frames / sample_rate
        return pcm_data, sample_rate, duration


#%% Representation of a frame audio data
class Frame(object):
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


#%% Frame generator: generates audio frames from PCM audio data (yields frames of the requested duration). Inputs: desire frame duration in ms, the PCM data, the sample rate.
def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset : offset + n], timestamp, duration)
        timestamp += duration
        offset += n


# %% PCM aud>io data generator to filter out non-voiced audio frames. Inputs:sample_rate frame_duration_ms, padding_duration_ms,instance of webrtcvad.Vad,frames. Returns: generator that yields PCM audio data.
def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_frames = []

    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)
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
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                yield b"".join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []

    if voiced_frames:
        yield b"".join([f.bytes for f in voiced_frames])


#%% Segment generator that will return the segment of byte data for the audio, but also and its metadata. Inputs: .wav file. Returns: tuple of audio segments, sample_rate, audio_length.
def vad_segment_generator(wav_file, aggressiveness):
    # print("Caught the wav file @: %s" % (wav_file))
    audio, sample_rate, audio_length = read_wave(wav_file)
    assert sample_rate == 16000, "Only 16000Hz input WAV files are supported for now!"
    vad = webrtcvad.Vad(int(aggressiveness))
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    segments = vad_collector(sample_rate, 30, 300, vad, frames)

    return segments, sample_rate, audio_length


#%% Function to load pre-trained model into the memory from DeepSpeech. Inputs: model, scorer. Returns: a list [DeepSpeech Object, Model Load Time, Scorer Load Time].
def load_model(models, scorer):
    model_load_start = timer()
    ds = Model(models)
    model_load_end = timer() - model_load_start
    # print("Loaded model in %0.3fs." % (model_load_end))

    scorer_load_start = timer()
    ds.enableExternalScorer(scorer)
    scorer_load_end = timer() - scorer_load_start
    # print("Loaded external scorer in %0.3fs." % (scorer_load_end))

    return [ds, model_load_end, scorer_load_end]


# %% Function to resolve directory path for the models. Input: path. Returns: a tuple containing each of the model files (pb, scorer).
def resolve_models(dir_name):
    pb: str = glob.glob(dir_name + "/*.pbmm")[0]
    # print("Found Model: %s" % pb)

    scorer: str = glob.glob(dir_name + "/*.scorer")[0]
    # print("Found scorer: %s" % scorer)

    return pb, scorer


#%% Function to transcript audio segments. Input: Deepspeech object, audio, sample_rate. Returns: a list [Inference, Inference Time, Audio Length].
def stt(ds, audio, fs):
    inference_time = 0.0
    # Run Deepspeech
    # print("Running inference...")
    inference_start = timer()
    output = ds.stt(audio)
    inference_end = timer() - inference_start
    inference_time += inference_end
    # print(
    #     "Inference took %0.3fs for %0.3fs audio file." % (inference_end, audio_length)
    # )

    return [output, inference_time]

