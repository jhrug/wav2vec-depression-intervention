import os  # for os.walk
import pathlib  # for path manipulations
import pandas as pd
import librosa  # loading to wav
import torchaudio  # backup load method
import librosa.display  # examining sound files
import numpy as np  # for daicwoz signal arrays
import soundfile as sf  # reading and writing sound files
from pathlib import Path
from datasets import Dataset, Audio, Features, Value
from transformers import AutoFeatureExtractor


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
    # check if it exists and is actually the parent folder
    if not parent_path.is_dir():
        raise NotADirectoryError(f"Parent directory not found: {parent_path}")
    if parent_path not in audio_path.parents:
        raise ValueError(f"Parent directory not found: {parent_path}")

    # obtain encoding name
    ext = audio_path.suffix.lower()
    # print(f"Detected extension: {ext}")

    # try with soundfile, else torchaudio
    try:
        # store daicwoz signal in compressed file as float32 array, save original sampling rate
        data, orig_sr = sf.read(audio_path, dtype='float32')

        # convert to mono if needed
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
            print("Converted stereo to mono (soundfile).")

    except Exception:
        # try same with torchaudio
        try:
            waveform, orig_sr = torchaudio.load(audio_path)

            # convert to mono
            data = waveform.mean(dim=0).numpy()

            print("Loaded with torchaudio.")
        # to implement later: librosa backup
        # data, orig_sr = librosa.load(audio_path, sr=None, dtype='float32')

        # if sf and torchaudio failed, raise
        except Exception as exc:
            raise RuntimeError(f"Could not read audio file: {exc}")

    # print original sample rate
    # print(f"Original sample rate: {orig_sr} Hz")

    if orig_sr != target_sr:
        # resample using librosa
        data = librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
        print(f"Resampled to: {target_sr} Hz")
    #
    # else:
    #    print("No resampling needed (original rate matches target).")
    return np.asarray(data, dtype=np.float32)


def obtain_participant_transcript_df(
        full_transcript_csv_path: str
) -> pd.DataFrame:
    # read csv and save to pandas dataframe
    transcript_df = pd.read_csv(full_transcript_csv_path, sep='\t')

    # create boolean mask for all rows that specify an utterance of Ellie, for which then ext utterance is also Ellie's
    boolean = (transcript_df['speaker'] == 'Ellie') & (transcript_df['speaker'].shift(-1) == 'Ellie')

    # drop all rows for which boolean mask is False
    df_filtered = transcript_df[~boolean].reset_index(drop=True)

    # set start time as end time of previous utterance (to include thinking pauses)
    df_filtered.loc[1:, 'start_time'] = df_filtered['stop_time'].shift(1)

    # drop all rows where Ellie speaks
    df_filtered = df_filtered[df_filtered['speaker'] != 'Ellie']

    return df_filtered


def load_participant_speech(
        orig_wav_path: str,
        full_transcript_csv_path: str
) -> np.ndarray:
    # same for all DAIC-WOZ data
    sr = 16000

    # load original wav as float32 array
    array = load_and_resample(orig_wav_path, sr)
    # print(array)

    import pandas as pd
    df = obtain_participant_transcript_df(full_transcript_csv_path)

    # initialize segment list
    segments = []

    for index, row in df.iterrows():
        # save start and end times for sample
        start_segment = int(row['start_time'] * sr)
        end_segment = int(row['stop_time'] * sr)

        # slice original array at start and end time
        audio_segment = array[start_segment:end_segment]

        # add to segments list
        segments.append(audio_segment)

    full_audio = np.concatenate(segments)

    # print(f"Original length: {len(array)} samples")
    # print(f"New length: {len(full_audio)} samples")

    return full_audio


def save_waveform(
        orig_audio_path: str,
        waveform: np.ndarray,
        sampling_rate: int,
        target_dir: str
) -> pathlib.Path:
    """writes .wav file to specified location from numpy float32 array

    Args: str specifying full original audio file path,
        numpy float 32 array specifying audio signal,
        target sampling rate,
        str specifying full path to directory in which to save .wav file

    Returns: path to .wav file
        """

    # turn string into Path object
    full_orig_audio_path = Path(orig_audio_path)

    # store original file name without file extension
    file_name = full_orig_audio_path.stem

    # store target parent directory
    out_dir = pathlib.Path(target_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # add _participant_only
    new_name = f"{file_name}_participant_only"

    # store target file name
    out_path = out_dir / new_name

    # out_path = Path(f"{out_path.stem}_cleaned{out_path.suffix}")
    out_path = out_path.with_suffix(".wav")

    # raise error if not .wav
    if out_path.suffix.lower() != ".wav":
        raise ValueError("Filename must end with '.wav'")

    # raise error if numpy array is of correct data type
    if waveform.dtype != np.float32:
        print("Warning: Waveform is not float32. Converted to float32")
        waveform = waveform.astype(np.float32)

    # NOTE TO SELF: check if downstream analysis actually expects PCM_16

    # write numpy array to target file
    sf.write(file=out_path, data=waveform, samplerate=sampling_rate, subtype="PCM_16")

    # print(f"WAV saved to: {out_path}")
    return out_path


def wav_participant_only(
        orig_wav_path: str,
        full_transcript_csv_path: str,
        target_dir: str
) -> pathlib.Path:
    # will be same for all DAIC-WOZ data
    sr = 16000

    # create array with only participant speech
    waveform = load_participant_speech(orig_wav_path, full_transcript_csv_path)

    # write array to wav file
    path = save_waveform(orig_wav_path, waveform, sr, target_dir)

    return path


def save_segment(
        orig_audio_path: str,
        waveform: np.ndarray,
        sampling_rate: int,
        segment_number: int,
        target_dir: str
) -> pathlib.Path:
    """modified version of save_waveform that writes .wav file to specified location from numpy float32 array with name specifying the number of the segment

    Args: str specifying full original audio file path,
        numpy float 32 array specifying audio signal,
        target sampling rate,
        str specifying full path to directory in which to save .wav file

    Returns: path to .wav file
        """

    # turn string into Path object
    full_orig_audio_path = Path(orig_audio_path)

    # store original file name without file extension
    file_name = full_orig_audio_path.stem

    # store target parent directory
    out_dir = pathlib.Path(target_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # add _participant_only
    new_name = f"{file_name}_segment_{segment_number}"

    # store target file name
    out_path = out_dir / new_name

    # out_path = Path(f"{out_path.stem}_cleaned{out_path.suffix}")
    out_path = out_path.with_suffix(".wav")

    # raise error if not .wav
    if out_path.suffix.lower() != ".wav":
        raise ValueError("Filename must end with '.wav'")

    # raise error if numpy array is of correct data type
    if waveform.dtype != np.float32:
        print("Warning: Waveform is not float32. Converted to float32")
        waveform = waveform.astype(np.float32)

    # NOTE TO SELF: check if downstream analysis actually expects PCM_16

    # write numpy array to target file
    sf.write(file=out_path, data=waveform, samplerate=sampling_rate, subtype="PCM_16")

    # print(f"WAV saved to: {out_path}")
    return out_path


def construct_segments_matrix(
        participant_1darray: np.ndarray,
        segment_length_seconds: int | float,
        sampling_rate: int
) -> np.ndarray:
    if type(segment_length_seconds) != int:
        print("Warning: segment_length_seconds is not an integer")
    # calculate length of segments in terms of samples
    samples_per_segment = segment_length_seconds * sampling_rate

    # discard tail
    num_segments = len(participant_1darray) // samples_per_segment
    print("number of segments: ", num_segments)

    # determine cut-off point
    truncated_length = num_segments * samples_per_segment
    # print("truncated length: ", truncated_length)

    # cut tail
    remainder = participant_1darray[:truncated_length]
    # print("remainder: ", remainder)

    # reshape into 2D array (segment number as added dimension)
    segments_matrix = remainder.reshape(num_segments, samples_per_segment)
    # print("segments_matrix: ", segments_matrix)

    return segments_matrix


def part_number_from_wav_path(orig_wav_path: str) -> int:
    """Extracts XXX from 'XXX_AUDIO.wav'."""
    # .stem gets 'XXX_AUDIO' (removes .wav)
    # .split('_')[0] gets the part before the first underscore
    split = pathlib.Path(orig_wav_path).stem.split("_")
    return int(split[0])


def sample_one_participant(
        orig_wav_path: str,
        full_transcript_csv_path: str,
        segment_length_seconds: int,
        target_dir: str,
        total_phq_score: int,
        participant_gender: str,
) -> pd.DataFrame:
    sr = 16000

    # save participant number as int
    participant_number = part_number_from_wav_path(orig_wav_path)
    print(f"participant_number: {participant_number}")

    # convert interview to participant-only signal as float32 ndarray
    participant_waveform = load_participant_speech(orig_wav_path, full_transcript_csv_path)

    # convert participant signal to segment matrix
    participant_matrix = construct_segments_matrix(participant_waveform, segment_length_seconds, sr)

    # pick one random segment
    segment_number = np.random.randint(0, len(participant_matrix))
    segment = participant_matrix[segment_number]
    print(f"randomly selected segment {segment_number} out of {len(participant_matrix)}")

    # save segment as wav and store path
    path = save_segment(orig_wav_path, segment, sr, segment_number, target_dir)

    # create row for segment
    row = {
        "participant": participant_number,
        "source_file": orig_wav_path,
        "phq_score": total_phq_score,
        "gender": participant_gender,
        "segment_number": segment_number,
        "audio": {
            "path": str(path),
            "array": segment,
            "sampling_rate": sr
        }
    }

    df = pd.DataFrame.from_records([row])
    return df


def segment_one_participant(
        orig_wav_path: str,
        full_transcript_csv_path: str,
        segment_length_seconds: int,
        target_dir: str,
        total_phq_score: int,
        participant_gender: str,
        participant_split: str
) -> pd.DataFrame:
    #""

    sr = 16000

    # save participant number as int
    participant_number = part_number_from_wav_path(orig_wav_path)
    # print(f"participant_number: {participant_number}")

    # convert interview to participant-only signal as float32 ndarray
    participant_waveform = load_participant_speech(orig_wav_path, full_transcript_csv_path)
    # print("participant_waveform: ", participant_waveform)

    # convert participant signal to segment matrix
    participant_matrix = construct_segments_matrix(participant_waveform, segment_length_seconds, sr)
    # print("participant_matrix: ", participant_matrix)

    # initialize dataset
    dataset_rows = []

    # loop through segment matrix
    for segment_number, segment in enumerate(participant_matrix):
        # save segment as wav and store path
        path = save_segment(orig_wav_path, segment, sr, segment_number, target_dir)
        # print(f"path: {path}")

        # create row for segment
        row = {
            "participant": participant_number,
            "source_file": orig_wav_path,
            "phq_score": total_phq_score,
            "gender": participant_gender,
            "split": participant_split,
            "segment_number": segment_number,
            "audio": {
                "path": str(path),
                "array": segment,
                "sampling_rate": sr
            }
        }
        # print("row: ", row)

        # append row to dataset object
        dataset_rows.append(row)
        # print("dataset_rows: ", dataset_rows)

        # construct dataset from list of dictionaries
        df = pd.DataFrame.from_records(dataset_rows)
        # print(df.head())

    if dataset_rows == []:
        return pd.DataFrame()

    return df


def is_full_interview(
        filename: str
) -> bool:
    """
    Returns True if the filename is of the form XXX_AUDIO.wav, where X are digits (0‑9), as in DAIC-WOZ. Must match exactly, otherwise false.
    """

    # must end with _AUDIO.wav
    if not filename.endswith(r"_AUDIO.wav"):
        return False

    # store everything before _AUDIO.wav
    prefix = filename[:-len(r"_AUDIO.wav")]
    # print("prefix: ", prefix)
    # print("len(prefix): ", len(prefix))
    # print("prefix.isdigit() = ", prefix.isdigit())

    # that must be three characters long
    if len(prefix) != 3:
        return False
    return prefix.isdigit()


def extract_lookup(
        full_dev_csv_path: str,
        full_train_csv_path: str,
        full_test_csv_path: str
) -> pd.DataFrame:
    #"takes csv files of dev and train and converts to DataFrame - WARNING NaNs in single PHQ questions will cause ints (0,1,2,3) to be converted to float"

    # load dev data
    phq_df_dev = pd.read_csv(full_dev_csv_path)
    phq_df_dev['split'] = 'dev'
    # print(phq_df_dev.head())

    # load train data
    phq_df_train = pd.read_csv(full_train_csv_path)
    phq_df_train['split'] = 'train'
    # print(phq_df_train.head())

    # load test data
    phq_df_test = pd.read_csv(full_test_csv_path)
    phq_df_test['split'] = 'test'
    phq_df_test = phq_df_test.rename(columns={'PHQ_Score': 'PHQ8_Score'})

    # fuse the two
    phq_df = pd.concat([phq_df_dev, phq_df_train, phq_df_test], ignore_index=True)
    # print(phq_df.head())

    return phq_df


def get_transcript_path(
        participant_number: int,
        root_dir: str
) -> Path:
    # ensure root is a Path
    root_path = Path(root_dir)

    # turn number into string
    pno_str = str(participant_number)

    # build expected filename
    filename = f"{pno_str}_TRANSCRIPT.csv"

    # full path to file
    transcript_path = root_path / filename

    # raise error if file does not exist
    if not transcript_path.is_file():
        raise FileNotFoundError(
            f"Transcript file not found: {transcript_path}"
        )
    return transcript_path


def look_up_phq(
        df: pd.DataFrame,
        participant_number: int
) -> int:
    key_col = "Participant_ID"
    value_col = "PHQ8_Score"

    # check if column names exist
    if key_col not in df.columns:
        raise KeyError(f"Column '{key_col}' not found in DataFrame")
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' not found in DataFrame")

    # locate matching rows
    mask = df[key_col] == participant_number
    matches = df.loc[mask, value_col]

    # raise error if no matching rows
    if matches.empty:
        raise ValueError(
            f"participant {participant_number} not found in column '{key_col}'"
        )

    # raise error if same participant appears twice
    if len(matches) > 1:
        raise ValueError(
            f"Multiple rows ({len(matches)}) found for participant {participant_number} in column '{key_col}'"
        )

    return matches.iloc[0]


def look_up_gender(
        df: pd.DataFrame,
        participant_number: int
) -> str:
    key_col = "Participant_ID"
    value_col = "Gender"

    # check if column names exist
    if key_col not in df.columns:
        raise KeyError(f"Column '{key_col}' not found in DataFrame")
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' not found in DataFrame")

    # locate matching rows
    mask = df[key_col] == participant_number
    matches = df.loc[mask, value_col]

    # raise error if no matching rows
    if matches.empty:
        raise ValueError(
            f"participant {participant_number} not found in column '{key_col}'"
        )

    # raise error if same participant appears twice
    if len(matches) > 1:
        raise ValueError(
            f"Multiple rows ({len(matches)}) found for participant {participant_number} in column '{key_col}'"
        )

    # store value as in original csv (0 or 1)
    binary = matches.iloc[0]

    # print("Gender data type is ", type(binary))

    # print("Gender: ", binary)
    if binary == 0:
        return str("feminine")
    if binary == 1:
        return "masculine"
    else:
        print("Oh no! We expected gender to be binary??")
        raise ValueError("look_up_gender(df, participant_no) assumes gender is 1 or 0")
        return None


def look_up_split(
        df: pd.DataFrame,
        participant_number: int
) -> str:
    key_col = "Participant_ID"
    value_col = "split"

    # check if column names exist
    if key_col not in df.columns:
        raise KeyError(f"Column '{key_col}' not found in DataFrame")
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' not found in DataFrame")

    # locate matching rows
    mask = df[key_col] == participant_number
    matches = df.loc[mask, value_col]

    # raise error if no matching rows
    if matches.empty:
        raise ValueError(
            f"participant {participant_number} not found in column '{key_col}'"
        )

    # raise error if same participant appears twice
    if len(matches) > 1:
        raise ValueError(
            f"Multiple rows ({len(matches)}) found for participant {participant_number} in column '{key_col}'"
        )

    return matches.iloc[0]


def make_segments_dir(orig_wav_path: str) -> Path:
    """
    Return
    :param orig_wav_path:
    :return: Path object pointing to a new directory called "<wav-stem>_segments" that lives in same directory as the root_wav_file)
    """

    wav_path = Path(orig_wav_path).resolve()
    segments_dir = wav_path.parent / f"{wav_path.stem}_180s_segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    return segments_dir


def construct_woz_dataset(
        root_dir_path: str,
        dev_csv_path: str,
        train_csv_path: str,
        test_csv_path: str,
        segment_length_seconds: int,
):
    #"Constructs full pandas dataset from root directory and csv files - WARNING: mostly assumes one full audio and transcript directory per participant"

    # get phq and gender data
    lookup_df = extract_lookup(dev_csv_path, train_csv_path, test_csv_path)

    # initialize dataframe
    dataset = pd.DataFrame(columns=["participant", "source_file", "phq_score", "audio"])

    # iteratively walk through all subdirectories
    for root, dirs, files in os.walk(root_dir_path):

        # read file names
        for filename in files:

            # if it is one of the audio files
            if is_full_interview(filename):

                # append filename to root path (stored as Path object)
                filepath = Path(root) / filename
                filepath = str(filepath)

                # store participant number as int
                participant_number = part_number_from_wav_path(filepath)

                # create directory for saving segments
                segments_dir = str(make_segments_dir(filepath))

                # find transcript path for participant
                # WARNING: assumes transcript is in same directory as audio
                transcript_path = get_transcript_path(participant_number, root)
                transcript_path = str(transcript_path)

                # find phq score for participant
                phq_score = look_up_phq(lookup_df, participant_number)

                # find gender for participant
                gender = look_up_gender(lookup_df, participant_number)

                # find split for participant
                split = look_up_split(lookup_df, participant_number)

                # create participant_only wav
                part_only_path = wav_participant_only(filepath, transcript_path, root)

                # construct dataframe for participant
                df_participant = segment_one_participant(filepath, transcript_path, segment_length_seconds,
                                                         segments_dir, phq_score, gender, split)

                if df_participant.empty:
                    print(
                        f"Skipping participant {participant_number}: insufficient audio for any segment of {segment_length_seconds} s")
                    continue

                # add participant to dataset
                dataset = pd.concat([dataset, df_participant], ignore_index=True)

                print(f"participant {participant_number} added to dataset. Dataset now has {dataset.shape[0]} rows")

    # print("final dataset: ", dataset.head())
    return dataset


if __name__ == "__main__":
    df = construct_woz_dataset(
        r"path\to\root\directory\to\scan\for\participant\audios",
        r"full\path\for\dev_split_Depression_AVEC2017.csv",
        r"full\path\for\train_split_Depression_AVEC2017.csv",
        r"full\path\for\full_test_split.csv",
        20)

    features = Features({
        "participant": Value("string"),
        "source_file": Value("string"),
        "phq_score": Value("float32"),
        "audio": Audio(sampling_rate=16000),
        "gender": Value("string"),
        "split": Value("string"),
        "segment_number": Value("int64")
    })

    # WARNING: prone to memory errors - split df in parts and cast separately if needed
    ds = Dataset.from_pandas(df, features=features, preserve_index=False)

    # LOAD NATIVE W2V2 PREPROCESSOR
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    # PREPARE PREPROCESS FUNCTION FOR MAP CALL
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

    data = ds.map(preprocess_function)

    data.save_to_disk("sample_name_woz")