"""PANNs CNN14 at 16 kHz (Kong et al. 2020, github.com/qiuqiangkong/audioset_tagging_cnn,
MIT licence), trimmed to inference + fine-tuning. Pretrained on AudioSet (mAP 0.438)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import LogmelFilterBank, Spectrogram

CKPT_URL = "https://zenodo.org/api/records/3987831/files/Cnn14_16k_mAP=0.438.pth/content"
LABELS_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn1, self.bn2 = nn.BatchNorm2d(cout), nn.BatchNorm2d(cout)

    def forward(self, x, pool=(2, 2)):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, pool)


class Cnn14(nn.Module):
    def __init__(self, classes_num=527):
        super().__init__()
        self.spectrogram_extractor = Spectrogram(n_fft=512, hop_length=160, win_length=512, window="hann",
                                                 center=True, pad_mode="reflect", freeze_parameters=True)
        self.logmel_extractor = LogmelFilterBank(sr=16000, n_fft=512, n_mels=64, fmin=50, fmax=8000,
                                                 ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True)
        self.bn0 = nn.BatchNorm2d(64)
        ch = [1, 64, 128, 256, 512, 1024, 2048]
        for i in range(6):
            setattr(self, f"conv_block{i + 1}", ConvBlock(ch[i], ch[i + 1]))
        self.fc1 = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, classes_num)

    def embed(self, wav, specaug=None):
        with torch.autocast(wav.device.type, enabled=False):          # STFT + log in fp32
            x = self.logmel_extractor(self.spectrogram_extractor(wav.float()))  # (B, 1, T, 64)
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)
        if specaug is not None:
            x = specaug(x)
        for i in range(6):
            x = getattr(self, f"conv_block{i + 1}")(x, (2, 2) if i < 5 else (1, 1))
            x = F.dropout(x, 0.2, self.training)
        x = x.mean(dim=3)
        x = x.max(dim=2)[0] + x.mean(dim=2)
        x = F.dropout(x, 0.5, self.training)
        return F.dropout(F.relu_(self.fc1(x)), 0.5, self.training)

    def forward(self, wav):
        return torch.sigmoid(self.fc_audioset(self.embed(wav)))


def load_pretrained(path) -> Cnn14:
    m = Cnn14()
    sd = torch.load(path, map_location="cpu", weights_only=False)["model"]
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing, missing
    assert all(k.startswith("spec_augmenter") for k in unexpected), unexpected
    return m


def fetch(work="/content/work"):
    """Download the checkpoint and AudioSet label order once. Returns (ckpt path, MIDs in order)."""
    import subprocess
    from pathlib import Path

    import pandas as pd
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    ck, lab = work / "cnn14_16k.pth", work / "class_labels_indices.csv"
    if not ck.exists():
        subprocess.run(f"curl -sSfL -o {ck} '{CKPT_URL}'", shell=True, check=True)
    if not lab.exists():
        subprocess.run(f"curl -sSfL -o {lab} {LABELS_URL}", shell=True, check=True)
    return ck, list(pd.read_csv(lab).mid)
