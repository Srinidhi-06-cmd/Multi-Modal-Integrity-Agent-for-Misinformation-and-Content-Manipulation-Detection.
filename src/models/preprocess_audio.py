import librosa
import numpy as np

def extract_mfcc(file_path, n_mfcc=40, max_len=130):
    """
    Extract MFCC features from an audio file.
    Output shape: (n_mfcc, max_len)
    """
    audio, sr = librosa.load(file_path, sr=22050)

    

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # Padding or truncating
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc
