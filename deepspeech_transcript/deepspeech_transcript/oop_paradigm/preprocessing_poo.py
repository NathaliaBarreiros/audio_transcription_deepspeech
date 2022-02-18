import contextlib
import wave
import collections
import webrtcvad
from typing import Generator, List, Dict, Tuple
from deepspeech import Model
from timeit import default_timer as timer
import glob
import numpy as np


class Frame():
    """
    Class Frame represents a frame of audio data.
    """

    def __init__(self, bytes: bytes, timestamp: float, duration: float) -> None:
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


class Preprocessing:
    """
    Class Preprocessing represents the pre-processing work to handle the .wav audio file. It creates all the functions necessary to convert a .wav file into segments of voiced PCM audio data that DeepSpeech can process.
    """

    def __init__(self, path: str, frame_duration_ms: int, padding_duration_ms: int, aggressiveness: int) -> None:
        self.path = path
        self.pcm_data = self.read_wave(path)["pcm_data"]
        self.sample_rate = self.read_wave(path)["sample_rate"]
        self.duration = self.read_wave(path)["duration"]
        self.frame_duration_ms = frame_duration_ms
        self.padding_duration_ms = padding_duration_ms
        self.vad = self.vad_segment_generator(path, aggressiveness)[3]
        self.frames = list(self.frame_generator(
            frame_duration_ms, self.pcm_data, self.sample_rate))
        self.aggressiveness = aggressiveness

    def read_wave(self, path: str) -> Dict:
        """
        Reads a .wav file
        Input: audio path.
        Returns: PCM audio data, sample rate, duration
        """
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

    def frame_generator(self, frame_duration_ms: int, pcm_data: bytes, sample_rate: int) -> Generator:
        """
        Generates audio frames from PCM audio data Inputs: desire frame duration in ms, the PCM data, the sample rate.
        Yields frames of the requested duration.
        """
        n = int(self.sample_rate * (frame_duration_ms/1000.0)*2)
        offset: int = 0
        timestamp: float = 0.0
        duration: float = (float(n)/sample_rate) / 2.0
        while offset + n < len(pcm_data):
            yield Frame(self.pcm_data[offset: offset + n], timestamp, duration)
            timestamp += duration
            offset += n

    def vad_collector(self, sample_rate: int, frame_duration_ms: int, padding_duration_ms: int, vad: webrtcvad.Vad, frames: List[Frame]) -> Generator:
        """
        PCM audio data generator to filter out non-voiced audio frames.
        Inputs:sample_rate frame_duration_ms, padding_duration_ms,instance of webrtcvad.Vad,frames.
        Returns: generator that yields PCM audio data.
        """
        num_padding_frames = int(padding_duration_ms/frame_duration_ms)
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
                num_unvoiced = len(
                    [f for f, speech in ring_buffer if not speech])
                if num_unvoiced > 0.9*ring_buffer.maxlen:
                    triggered = False
                    yield b"".join([f.bytes for f in voiced_frames])
                    ring_buffer.clear()
                    voiced_frames = []
        if voiced_frames:
            yield b"".join([f.bytes for f in voiced_frames])

    def vad_segment_generator(self, path: str, aggressiveness: int) -> List:
        """
        Segment generator that will return the segment of byte data for the audio, but also its metadata.
        Inputs: .wav file path.
        Returns: tuple of audio segments, sample_rate, audio_length, vad.
        """
        read_instance = self.read_wave(path)
        assert read_instance["sample_rate"] == 16000, "Only 16000Hz input WAV files are supported for now!"
        frame_instance = self.frame_generator(
            30, read_instance["pcm_data"], read_instance["sample_rate"])
        frame_instance = list(frame_instance)
        vad = webrtcvad.Vad(int(aggressiveness))
        segments = self.vad_collector(
            read_instance["sample_rate"], 30, 300, vad, frame_instance)

        return [segments,  read_instance["sample_rate"], read_instance["duration"], vad]


class DeepSpeechModel:
    """
    Class DeepSpeechModel represents the model path resolution in order to obtain model's files, the load of pre-trained models and the transcription of the audio segments using the loaded models on memory from DeepSpeech.
    """

    def __init__(self, dir_name: str) -> None:
        self.dir_name = dir_name
        self.models = self.resolve_models_paths()[0]
        self.scorer = self.resolve_models_paths()[1]
        self.ds = self.load_models(self.models, self.scorer)[0]

    def resolve_models_paths(self) -> list[str, str]:
        """
        Function to resolve directory path for the models.
        Input: path.
        Returns: a list containing each of the model files (pb, scorer).
        """
        pb: str = glob.glob(self.dir_name+"/*.pbmm")[0]
        scorer: str = glob.glob(self.dir_name+"/*.scorer")[0]
        return [pb, scorer]

    def load_models(self, models: str, scorer: str) -> list[Model, float, float]:
        """
        Function to load pre-trained model into the memory from DeepSpeech.
        Inputs: model, scorer.
        Returns: a list [DeepSpeech Object, Model Load Time, Scorer Load Time].
        """
        model_load_start = timer()
        ds = Model(models)
        model_load_end = timer() - model_load_start

        scorer_load_start = timer()
        ds.enableExternalScorer(scorer)
        scorer_load_end = timer() - scorer_load_start

        return [ds, model_load_end, scorer_load_end]

    def transcript_audio_segments(self, ds: Model, audio_stt: np.ndarray) -> list[str, float]:
        """
        Function to transcript audio segments.
        Input: Deepspeech object, audio as np.ndarray type.
        Returns: a list [Inference, Inference Time].
        """
        inference_time: float = 0.0
        inference_start = timer()
        output: str = ds.stt(audio_stt)

        inference_end = timer() - inference_start
        inference_time += inference_end

        return [output, inference_time]
