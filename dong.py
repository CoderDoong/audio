# -*- coding: utf-8 -*-

# 安装依赖项
# !pip install git+https://github.com/microsoft/CLAP.git
# !pip install audiomentations
# !pip install torchaudio

import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
import torch.nn as nn
import pandas as pd
import torchaudio
from sklearn.metrics import accuracy_score

# ESC-50 数据集类定义
class AudioDataset(Dataset):
    def __init__(self, root: str, download: bool = True):
        self.root = os.path.expanduser(root)
        if download:
            self.download()

    def __getitem__(self, index):
        raise NotImplementedError

    def download(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class ESC50(AudioDataset):
    base_folder = 'ESC-50-master'
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    filename = "ESC-50-master.zip"
    num_files_in_dir = 2000
    audio_dir = 'audio'
    label_col = 'category'
    file_col = 'filename'
    meta = {
        'filename': os.path.join('meta','esc50.csv'),
    }

    def __init__(self, root, reading_transformations: nn.Module = None, download: bool = True):
        super().__init__(root)
        self._load_meta()
        self.targets, self.audio_paths = [], []
        self.pre_transformations = reading_transformations
        print("Loading audio files")
        self.df['category'] = self.df['category'].str.replace('_',' ')

        for _, row in tqdm(self.df.iterrows()):
            file_path = os.path.join(self.root, self.base_folder, self.audio_dir, row[self.file_col])
            self.targets.append(row[self.label_col])
            self.audio_paths.append(file_path)

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        self.df = pd.read_csv(path)
        self.class_to_idx = {}
        self.classes = [x.replace('_',' ') for x in sorted(self.df[self.label_col].unique())]
        for i, category in enumerate(self.classes):
            self.class_to_idx[category] = i

    def __getitem__(self, index):
        file_path, target = self.audio_paths[index], self.targets[index]
        idx = torch.tensor(self.class_to_idx[target])
        one_hot_target = torch.zeros(len(self.classes)).scatter_(0, idx, 1).reshape(1,-1)
        return file_path, target, one_hot_target

    def __len__(self):
        return len(self.audio_paths)

    def download(self):
        download_url(self.url, self.root, self.filename)
        from zipfile import ZipFile
        with ZipFile(os.path.join(self.root, self.filename), 'r') as zip:
            zip.extractall(path=self.root)

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, TimeMask
# 定义使用 audiomentations 的增强方法
augmenter = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.010, p=1),  # 加入高斯噪声 噪声的振幅在 0.001 到 0.015
    TimeStretch(min_rate=0.8, max_rate=1.25, p=1),  # 时间拉伸 时间伸缩的倍数会在 0.8 到 1.25 之间随机选择
    PitchShift(min_semitones=-4, max_semitones=4, p=1),  # 音高变化 音高在 -4 到 4 个半音之间随机变化，负值降低音高，正值提升音高。
    TimeMask(min_band_part=0.1, max_band_part=0.3,p=1) # 时间遮罩 遮罩的时长在音频的 10% 到 30% 之间
])

import tempfile
import torchaudio
# 增强后的数据集类
class AudioDatasetWithAugmentation(ESC50):
    def __init__(self, root, augmentations=None, download=True):
        super().__init__(root, download=download)
        self.augmentations = augmentations

    def apply_augmentation(self, audio_tensor, sample_rate):
        if self.augmentations:
            audio_tensor = audio_tensor.numpy()
            augmented_audio = self.augmentations(samples=audio_tensor, sample_rate=sample_rate)
            return torch.tensor(augmented_audio)
        return audio_tensor

    def __getitem__(self, index):
        file_path, target, one_hot_target = super().__getitem__(index)
        waveform, sample_rate = torchaudio.load(file_path)
        # 应用数据增强
        waveform = self.apply_augmentation(waveform, sample_rate)
        # 创建一个临时文件来保存增强后的音频
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        torchaudio.save(temp_file.name, waveform, sample_rate)
        return temp_file.name, target, one_hot_target

# 使用 CLAP 模型和数据集
from msclap import CLAP
# 1. 加载ESC-50数据集
root_path = "/"
dataset = ESC50(root=root_path, download=True)
dataset_with_augmentation = AudioDatasetWithAugmentation(root=root_path, augmentations=augmenter, download=True)
prompt = 'this is the sound of '
y = [prompt + x for x in dataset.classes]  # 创建类别描述

# 2. 加载并初始化CLAP模型
clap_model = CLAP(version='2023', use_cuda=True)  # 使用GPU加速

# 3. 计算文本嵌入
print("Computing text embeddings...")
text_embeddings = clap_model.get_text_embeddings(y)

# 4. 计算音频嵌入并进行零样本分类
y_preds, y_labels = [], []
print("Computing audio embeddings with audiomentations and performing zeroshot classification...")
for i in tqdm(range(len(dataset_with_augmentation))):  # tqdm用于显示进度条
    waveform, _, one_hot_target = dataset_with_augmentation.__getitem__(i)
    audio_embeddings = clap_model.get_audio_embeddings([waveform], resample=True)
    similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)  # 计算音频与文本嵌入的相似度
    y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()  # 使用softmax将相似度转换为概率
    y_preds.append(y_pred)
    y_labels.append(one_hot_target.detach().cpu().numpy())

# 5. 计算分类准确率
y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)  # 合并所有样本的标签和预测值
acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))  # 计算准确率
print('ESC50 Accuracy with audiomentations: {:.2f}%'.format(acc * 100))

# 6. 计算宏观 F1-score
from sklearn.metrics import f1_score
f1 = f1_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1), average='macro')
print('F1 Score: {:.2f}'.format(f1))

# 7. 计算 mAP
from sklearn.metrics import average_precision_score
mAP = average_precision_score(y_labels, y_preds, average='macro')
print('mAP: {:.2f}'.format(mAP))

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 计算混淆矩阵
cm = confusion_matrix(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))

# 正规化混淆矩阵
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# 可视化混淆矩阵
plt.figure(figsize=(10, 7))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="coolwarm", linewidths=1, linecolor='black')

# 调整标签和字体
plt.title('Confusion Matrix', fontsize=15)
plt.xlabel('Predicted Labels', fontsize=12)
plt.ylabel('True Labels', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()

# 计算混淆矩阵
cm = confusion_matrix(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))

# 每个类别的准确率
class_accuracy = cm.diagonal() / cm.sum(axis=1)
for i, class_name in enumerate(dataset.classes):
    print(f"Accuracy for {class_name}: {class_accuracy[i]:.2f}")