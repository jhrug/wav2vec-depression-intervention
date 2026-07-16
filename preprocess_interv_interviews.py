import os #for os.walk
import re
import pathlib # for path manipulations
import pandas as pd
import librosa # loading to wav
import torchaudio # backup load method
import librosa.display # examining sound files
import numpy as np # for daicwoz signal arrays
import soundfile as sf # reading and writing sound files
from collections import Counter, defaultdict
from pathlib import Path
from datasets import Dataset, Audio, Features, Value, Sequence, load_from_disk, concatenate_datasets
from transformers import AutoFeatureExtractor, Wav2Vec2Config

ROOT_DIR = (r"full\path\to\directory\containing\all\journal\submissions")

def all_files_in_dir_asPath(
    root_dir_path: str,
) -> Path:
    #iteratively walk through all subdirectories
    for root, dirs, files in os.walk(root_dir_path):

        #read file names
        for filename in files:

            #return full path
            yield Path(root) / filename

def extensions(
        root_dir_path: str,
) -> list:
    extensions = []

    for file in all_files_in_dir_asPath(root_dir_path):
        extension = file.suffix.lower()

        if extension not in extensions:
            extensions.append(extension)

    return extensions

def is_home(stem: str) -> bool:
    return re.match(r"P\d{2}_Q1_d([1-9]|1[0-3])", stem)

def is_lab(stem: str) -> bool:
    return re.match(r"P\d{2}_Q1_(pre|post(1|2))", stem)

def is_pilot(stem: str) -> bool:
    return re.match(r"PP\d_Q1_(d[1-7]|(pre|post(1|2)))", stem)

def construct_row(
    participant_number: int,
    day: int,
    segment_number: int,
    path: pathlib.Path,
    array: np.ndarray,
    sr: int,
    long_enough: bool
) -> dict:
    row = {
        "participant": participant_number,
        "day": day,
        "segment_number": segment_number,
        "audio": {
            "path": str(path),
            "array": array,
            "sampling_rate": sr
        },
        "long_enough": long_enough
    }
    return row

def save_segment(
        segments_dir: pathlib.Path,
        participant_number: int,
        day: int,
        array: np.ndarray,
        sr: int,
        segment_number: int,
) -> pathlib.Path:

    #STORE SEGMENT PATH
    segment_path = segments_dir / f"P{participant_number}_d{day}_segment_{segment_number}"

    #ADD FILE EXTENSION
    segment_path = segment_path.with_suffix(".wav")

    #WRITE NUMPY ARRAY TO TARGET PATH
    sf.write(file=segment_path, data=array, samplerate=sr, subtype="PCM_16")

    #print(f"WAV saved to: {out_path}")
    return segment_path


def construct_segments_matrix(
        participant_1darray: np.ndarray,
        segment_length_seconds: int | float,
        sampling_rate: int
) -> np.ndarray:
    if type(segment_length_seconds) != int:
        print("Warning: segment_length_seconds is not an integer")

    # calculate length of segments in terms of samples
    samples_per_segment = segment_length_seconds * sampling_rate

    # calculate number of segments
    num_segments = len(participant_1darray) // samples_per_segment
    print("number of segments: ", num_segments)

    # determine cut-off point
    truncated_length = num_segments * samples_per_segment
    # print("truncated length: ", truncated_length)

    # discard tail
    remainder = participant_1darray[:truncated_length]
    # print("remainder: ", remainder)

    # reshape into 2D array (segment number as added dimension)
    segments_matrix = remainder.reshape(num_segments, samples_per_segment)
    # print("segments_matrix: ", segments_matrix)

    return segments_matrix


def construct_raw_home(
        root_dir_path: str,
        segment_length: int,
        sr: int
) -> pd.DataFrame:
    dataset = []

    for file in all_files_in_dir_asPath(root_dir_path):

        # ONLY DO RECORDINGS MADE AT HOME
        stem = file.stem
        if is_home(stem):

            # OBTAIN PARTICIPANT NUMBER AND DAY
            split = file.stem.split("_")
            participant_number = int(split[0][1:])
            day = int(split[2][1:])

            # CREATE DIRECTORY FOR SAVING SEGMENTS
            parent = file.parent
            segments_dir = parent / f"P{participant_number}_d{day}_segments"
            segments_dir.mkdir(parents=True, exist_ok=True)

            # LOAD AUDIO INTO ARRAY
            array = load_and_resample(file, sr)

            # CHECK IF AUDIO IS LONGER THAN SPECIFIED SEGMENT LENGTH
            long_enough = len(array) / sr > segment_length

            # IF ARRAYS TOO SHORT TO SLICE, SAVE ONLY THIS ONE
            if not long_enough:
                print("WARNING: Audio file shorter than specified segment length. Returned original audio array")

                # SAVE FILE TO SUBDIR
                segment_path = save_segment(segments_dir, participant_number, day, array, sr, 1)

                # CONSTRUCT ROW AND ADD TO DATASET
                row = construct_row(participant_number, day, 1, segment_path, array, sr, long_enough)
                dataset.append(row)

            # IF MULTIPLE SEGMENTS
            else:
                # CONSTRUCT MATRIX
                matrix = construct_segments_matrix(array, segment_length, sr)

                # ITERATE THROUGH MATRIX
                for segment_number, segment in enumerate(matrix):
                    # SAVE FILE TO SUBDIR
                    segment_path = save_segment(segments_dir, participant_number, day, segment, sr, segment_number)

                    # CONSTRUCT ROW AND ADD TO DATASET
                    row = construct_row(participant_number, day, segment_number, segment_path, segment, sr, long_enough)
                    dataset.append(row)

            # print(f"participant: {participant_number}, day: {day}, audio: {array}")
    return pd.DataFrame(dataset)

if __name__ == "__main__":
    df = construct_raw_home(ROOT_DIR, 20, 16000)

    features = Features({
        "participant": Value("int64"),
        "day": Value("int64"),
        "segment_number": Value("int64"),
        "audio": Audio(sampling_rate=16000),
        "long_enough": Value("bool")
    })

    ds = Dataset.from_pandas(df, features=features, preserve_index=False)

    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    filtered = ds.filter(lambda x: x["long_enough"])

    def preprocess_function(batch):
        # load audio values from torchcodec.AudioDecoder object
        audio_arrays = [x.get_all_samples().data for x in batch["audio"]]

        # perform wav2vec2-base feature extraction
        inputs = feature_extractor(
            audio_arrays, sampling_rate=feature_extractor.sampling_rate, max_length=32000, truncation=True
        )

        # additional manipulation needed, as wav2vec feature extractor returns a <class 'transformers.feature_extraction_utils.BatchFeature'> of shape (113, 1, 16000), while downstream it expects a list of 1D arrays whose length equals the batch size
        inter = inputs.get("input_values")
        tuple_inst = inter[0]
        tuple_squeezed = np.squeeze(tuple_inst, axis=1)
        list_of_arrays = [tuple_squeezed[i] for i in range(tuple_squeezed.shape[0])]
        dict_of_array_list = {"input_values": list_of_arrays}
        return dict_of_array_list

    filtered = filtered.map(preprocess_function, batched=True)

    filtered.save_to_disk("home_without_preds")