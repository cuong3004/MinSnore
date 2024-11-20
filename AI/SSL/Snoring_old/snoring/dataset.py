import torchaudio
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import torch
from torchaudio.transforms import Spectrogram, Resample, MelSpectrogram, MelScale, AmplitudeToDB
from torchaudio.transforms import TimeStretch, FrequencyMasking, TimeMasking
import numpy as np 
import librosa

class CusttomDataUrban:
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms
        self.pipeline = MyPipeline()

        label_list = sorted(list(set(df.iloc[:,-2])))
        self.label_dir = {k:v for v, k in enumerate(label_list)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = "/kaggle/input/urbansound8k/fold" + str(self.df["fold"].iloc[idx]) + "/" + self.df["slice_file_name"].iloc[idx]
        label = self.df["classID"].iloc[idx]

        wav, sr = torchaudio.load(path)

        target_sr = 44100  # Set your target sample rate here
        if sr != target_sr:
            wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
            sr = target_sr

        wav = torch.mean(wav, dim=0, keepdim=True)
        # pad 4 second
        tempData = torch.zeros([1, sr*4])
        if wav.numel() < sr*4: # if sample_rate < 160000
            tempData[:, :wav.numel()] = wav
        else:
            tempData = wav[:, :sr*4] # else sample_rate 160000
        wav=tempData
        # return {"data": wav, "label": label}
        
        features = self.pipeline(wav)
        return {"data": features, "label": label}
    

class CusttomDataSnor:
    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms
        self.pipeline = MyPipeline()
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df["Link"].iloc[idx]
        label = int(self.df["Label"].iloc[idx])

        wav, sr = torchaudio.load(path)

        target_sr = 44100  # Set your target sample rate here
        if sr != target_sr:
            wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
            sr = target_sr

        # convert to mono
        audio_mono = torch.mean(wav, dim=0, keepdim=True)
        
        # pad 4 second
        # tempData = torch.zeros([1, sr*4])
        # if audio_mono.numel() < sr*4: # if sample_rate < 160000
        #     tempData[:, :audio_mono.numel()] = audio_mono
        # else:
        #     tempData = audio_mono[:, :sr*4] # else sample_rate 160000
        # audio_mono=tempData
        # return {"data": wav, "label": label}
        
        features = self.pipeline(audio_mono)
        return {"data": features, "label": label}
    
class MyPipeline(torch.nn.Module):
    def __init__(
        self,
        input_freq=44100,
        stretch_factor=0.8,
    ):
        super().__init__()

        self.mel_spec = MelSpectrogram(
            sample_rate=input_freq,
                n_fft=1024,
                win_length=None,
                hop_length=512,
                center=True,
                pad_mode="reflect",
                power=2.0,
                norm='slaney',
                onesided=True,
                n_mels=128,
                mel_scale="htk",
            )
        
        self.spec_aug = torch.nn.Sequential(
            TimeStretch(stretch_factor, fixed_rate=True),
            FrequencyMasking(freq_mask_param=80),
            TimeMasking(time_mask_param=80),
        )

        self.log_scale = AmplitudeToDB(top_db=80)
        
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mel_spec = self.mel_spec(waveform)

        # Apply SpecAugment
        # spec = self.spec_aug(spec)
        
        D_harmonic, D_percussive = librosa.decompose.hpss(mel_spec)
        
        
        

        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        log_D_harmonic = librosa.power_to_db(D_harmonic, ref=np.max)
        log_D_percussive = librosa.power_to_db(D_percussive, ref=np.max)
        
        
        
        
        # log_mel = self.log_scale(mel_spec)
        log_mel = torch.from_numpy(log_mel)
        log_D_harmonic = torch.from_numpy(log_D_harmonic)
        log_D_percussive = torch.from_numpy(log_D_percussive)

        return min_max_normalize(torch.concat((log_mel, log_D_harmonic,log_D_percussive), 0))   

def min_max_normalize(data, epsilon=1e-7):
    """
    Chuẩn hóa Min-Max với epsilon
    Args:
    - data: Mảng dữ liệu cần chuẩn hóa
    - epsilon: Giá trị epsilon để tránh chia cho 0
    Returns:
    - normalized_data: Mảng đã được chuẩn hóa
    """
    min_val = -80.0
    max_val = 0.0
    normalized_data = (data - min_val) / (max_val - min_val + epsilon)
    return normalized_data

# Example usage:
# data_to_normalize = [2, 5, 10, 7, 3]
# normalized_data = min_max_normalize(data_to_normalize)

# print("Original data:", data_to_normalize)
# print("Normalized data:", normalized_data)
