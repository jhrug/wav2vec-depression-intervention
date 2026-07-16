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

ROOT_DIR = (r"full\path\to\directory\containing\all\interview\files")

def all_files_in_dir_asPath(
    root_dir_path: str,
) -> Path:
    #iteratively walk through all subdirectories
    for root, dirs, files in os.walk(root_dir_path):

        #read file names
        for filename in files:

            #return full path
            yield Path(root) / filename

def all_files_in_dir_asStr(
    root_dir_path: str,
) -> Path:
    #iteratively walk through all subdirectories
    for root, dirs, files in os.walk(root_dir_path):

        #read file names
        for filename in files:

            #return full path
            yield str(Path(root) / filename)


def extensions(
        root_dir_path: str,
) -> list:
    extensions = []

    for file in all_files_in_dir_asPath(root_dir_path):
        extension = file.suffix.lower()

        if extension not in extensions:
            extensions.append(extension)

    return extensions

def is_lab(stem: str) -> bool:
    return re.match(r"P\d{2}_Q[2-4]_(pre|post(1|2))", stem)

def is_averaged(stem: str) -> bool:
    return re.match(r"P\d{2}_Q[2-4]_(pre|post(1|2))_D06", stem)

def is_pilot(stem: str) -> bool:
    return re.match(r"PP[1-3]_Q[2-4]_(pre|post(1|2))", stem)

def naming_convention_fails(root_dir_path: str) -> list:
    fails = []

    for file in all_files_in_dir_asPath(root_dir_path):
        stem = file.stem
        if not (is_averaged(stem) or is_lab(stem) or is_pilot(stem)):
            fails.append(stem)
    print(f"the following filenames do NOT match the naming convention: {fails}")
    print("all other files match the naming convention.")
    return fails

def load_and_resample(
        compressed_audio_path: str,
        target_sr: int
) -> np.ndarray:
    """load and resample audio from compressed audio

    Args: str specifying full audio file path
            (e.g.: "C:\\Users\\UserName\\Documents\\Testfile.m4a"),
        target sampling rate (e.g. 48000)
    Returns: numpy float32 array specifying audio signal"""

    # convert string to Path object
    audio_path = pathlib.Path(compressed_audio_path).resolve()
    # check if it exists
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # store parent directory as path object
    parent_path = audio_path.parent.resolve()

    # obtain encoding name
    ext = audio_path.suffix.lower()
    # print(f"Detected extension: {ext}")

    # try with soundfile, else torchaudio
    try:
        # store signal in compressed file as float32 array, save original sampling rate
        data, orig_sr = sf.read(audio_path, dtype='float32')
        print(f"loaded data with sf.read: {data}")

        # convert to mono (NOT THROUGH AVERAGING BUT MAX ENERGY SELECTION)
        if len(data.shape) > 1:
            print(f"len(data.shape) = {len(data.shape)} - len(data.shape) > 1 = {len(data.shape) > 1}")
            data = data[:, np.argmax(np.abs(data).mean(axis=0))]
            # data = np.mean(data, axis=1)
            # print("Converted stereo to mono (soundfile).")

    except Exception:
        # try same with torchaudio
        try:
            waveform, orig_sr = torchaudio.load(audio_path)
            print(f"loaded with torchaudio: waveform = {waveform}, orig_sr = {orig_sr}")

            # convert to mono (NOT THROUGH AVERAGING BUT MAX ENERGY SELECTION)
            data = waveform.argmax(waveform.abs().mean(dim=1))
            print(f"converted to mono through taking mean. Result: {data}")

        # if sf and torchaudio failed, raise
        except Exception as exc:
            raise RuntimeError(f"Could not read audio file: {exc}")

    if orig_sr != target_sr:
        # resample using librosa
        data = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
        print(f"Original sample rate: {orig_sr} Hz - resampled to: {target_sr} Hz")
    else:
        print("No resampling needed (original rate matches target).")

    print(f"resulting array: {np.asarray(data, dtype=np.float32)}")

    return np.asarray(data, dtype=np.float32)


def construct_row(
    participant_number: int,
    question_number: int,
    t: int,
    segment_number: int,
    path: pathlib.Path,
    array: np.ndarray,
    sr: int,
    long_enough: bool
) -> dict:
    row = {
        "participant": participant_number,
        "question_number": question_number,
        "t": t,
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
        question_number: int,
        t: int,
        array: np.ndarray,
        sr: int,
        segment_number: int,
) -> pathlib.Path:

    #STORE SEGMENT PATH
    segment_path = segments_dir / f"P{participant_number}_Q{question_number}_t{t}_segment_{segment_number}"

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


def construct_raw_mic(
        root_dir_path: str,
        segment_length: int,
        sr: int
) -> pd.DataFrame:
    grouped = defaultdict(list)

    # FIRST CREATE A LIST OF GROUPS TO ITERATE THROUGH
    for file in all_files_in_dir_asPath(root_dir_path):
        stem = file.stem

        if "PP" in stem:
            continue

        # OBTAIN PARTICIPANT NUMBER AND QUESTION
        split = file.stem.split("_")
        participant_number = int(split[0][1:])
        question_number = int(split[1][1:])

        if "post" in split[2]:
            t = split[2][4]
        else:
            t = 0

        grouped[(participant_number, t)].append((question_number, file))

    # INITIALIZE DATASET
    dataset = []

    for (participant_number, t), question_answers in grouped.items():

        # SORT BY QUESTION NUMBER
        question_answers.sort(key=lambda x: x[0])

        # LOAD AUDIOS INTO ARRAYS
        arrays = [load_and_resample(file, sr) for question_number, file in question_answers]

        # CONCATENATE ARRAYS INTO ONE
        array = np.concatenate(arrays)

        # SET QUESTION NUMBER TO NONE TO INDICATE CONCATENATED
        question_number = None

        # CREATE DIRECTORY FOR SAVING SEGMENTS
        parent = file.parent
        segments_dir = parent / f"P{participant_number}_segments"
        segments_dir.mkdir(parents=True, exist_ok=True)

        # CHECK IF AUDIO IS LONGER THAN SPECIFIED SEGMENT LENGTH
        long_enough = len(array) / sr > segment_length

        # IF ARRAYS TOO SHORT TO SLICE, SAVE ONLY THIS ONE
        if not long_enough:
            print("WARNING: Audio file shorter than specified segment length. Returned original audio array")

            # SAVE FILE TO SUBDIR
            segment_path = save_segment(segments_dir, participant_number, question_number, t, array, sr, 1)

            # CONSTRUCT ROW AND ADD TO DATASET
            row = construct_row(participant_number, question_number, t, 1, segment_path, array, sr, long_enough)
            dataset.append(row)

        # IF MULTIPLE SEGMENTS
        else:
            # CONSTRUCT MATRIX
            matrix = construct_segments_matrix(array, segment_length, sr)

            # ITERATE THROUGH MATRIX
            for segment_number, segment in enumerate(matrix):
                # SAVE FILE TO SUBDIR
                segment_path = save_segment(segments_dir, participant_number, question_number, t, segment, sr,
                                            segment_number)

                # CONSTRUCT ROW AND ADD TO DATASET
                row = construct_row(participant_number, question_number, t, segment_number, segment_path, segment, sr,
                                    long_enough)
                dataset.append(row)

        # print(f"participant: {participant_number}, day: {day}, audio: {array}")
    return pd.DataFrame(dataset)

if __name__ == "__main__":
    naming_convention_fails(ROOT_DIR)
    df = construct_raw_mic(ROOT_DIR, 20, 16000)

    # HOW MANY SESSIONS CONSIST ONLY OF ANSWERS TOO SHORT TO SAVE AS SEGMENT
    df.groupby(["participant", "t"])["long_enough"].any().value_counts()

    features = Features({
        "participant": Value("int64"),
        "question_number": Value("int64"),
        "t": Value("int64"),
        "segment_number": Value("int64"),
        "audio": Audio(sampling_rate=16000),
        "long_enough": Value("bool")
    })

    ds = Dataset.from_pandas(df, features=features, preserve_index=False)

    # KEEP ONLY ROWS FROM RECORDINGS THAT WERE 20 S OR LONGER
    filtered = ds.filter(lambda x: x["long_enough"])

    # LOAD FEATURE EXTRACTOR
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    # SET PREPROCESS FUNCTION FOR MAP
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

    filtered.save_to_disk("mic_without_preds")