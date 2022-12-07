from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split as tts
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class SynthDataset(Dataset):


    def __init__(self, pairs, train_mode=True, img_height=32, img_width=96, voc = None):
        super(SynthDataset, self).__init__()
        self.pairs = pairs
        if voc:
            self.chars = voc
        self.CHAR2LABEL = {char: i + 1 for i, char in enumerate(self.chars)}
        self.LABEL2CHAR = {label: char for char, label in self.CHAR2LABEL.items()}

        if train_mode:
            self.transform = transforms.Compose(
                [   transforms.ColorJitter(brightness=0.1, contrast=0.1, 
                                           saturation=0.1, hue=0.1),
                    transforms.RandomEqualize(),
                    transforms.RandomChoice([
                        transforms.RandomAdjustSharpness(sharpness_factor=0, p=0.33),
                        transforms.RandomAdjustSharpness(sharpness_factor=1, p=0.33),
                        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.33),
                    ]),
                    transforms.Grayscale(1),
                    transforms.Resize([img_height, img_width]),
                    transforms.RandomRotation(3),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Grayscale(1),
                    transforms.Resize([img_height, img_width]),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        pair = self.pairs[index]
        img = Image.open(pair[0])
        if self.transform is not None:
            img = self.transform(img)

        text = pair[1]
        target = [self.CHAR2LABEL[c] for c in text]
        target_length = [len(target)]
        target = torch.LongTensor(target)
        target_length = torch.LongTensor(target_length)

        return img, target, target_length


def collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths


def get_file_text(path):

    with open(path, "r") as file:
        data = file.read().rstrip()
    return data


def splitter(root, val_size=0.2, type = 'custom'):

    if type == 'news_labels':

        im_gen = Path(root).glob("**/*.png")
        images = sorted([i for i in im_gen if i.is_file()])
        txt_gen = Path(root).glob("**/*.txt")
        texts = sorted([i for i in txt_gen if i.is_file()])


        X, Y = np.arange(len(images)), np.arange(len(images))
        idxs_train, idxs_test, _, _ = tts(X, X, test_size=val_size, random_state=1111)

        train_im = [images[i] for i in idxs_train]
        train_txt = [texts[i] for i in idxs_train]
        test_im = [images[i] for i in idxs_test]
        test_txt = [texts[i] for i in idxs_test]


    elif type == 'custom':
        im_gen = Path(root).glob("**/*.jpg")
        images = sorted([i for i in im_gen if i.is_file()])


        X, Y = np.arange(len(images)), np.arange(len(images))
        idxs_train, idxs_test, _, _ = tts(X, X, test_size=val_size, random_state=1111)

        train_im = [images[i] for i in idxs_train]
        test_im = [images[i] for i in idxs_test]
        train_txt = []
        test_txt = []

        for f in train_im:
            label, _ = str(f.stem).split('_')
            train_txt.append(label)

        for f in test_im:
            label, _ = str(f.stem).split('_')
            test_txt.append(label)

    train_pairs = [(k, v) for k, v in list(zip(train_im, train_txt))]
    test_pairs = [(k, v) for k, v in list(zip(test_im, test_txt))]

    return (train_pairs, test_pairs)
