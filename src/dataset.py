from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split as tts
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class SynthDataset(Dataset):
    RUS_LETTERS = "АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя"
    NUMBERS = "0123456789"
    PUNCTUATION_MARKS = ' .,?:;—!<>-«»()[]*"'
    CHARS = RUS_LETTERS + NUMBERS + PUNCTUATION_MARKS
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, pairs, train_mode=True, img_height=32, img_width=100):
        super(SynthDataset, self).__init__()
        self.pairs = pairs

        if train_mode:
            self.transform = transforms.Compose(
                [
                    transforms.Grayscale(1),
                    transforms.Resize([img_height, img_width]),
                    transforms.RandomRotation(7),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Grayscale(1),
                    transforms.Resize([img_height, img_width]),
                    transforms.RandomRotation(7),
                    # transforms.RandomHorizontalFlip(),
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


def splitter(root, val_size=0.2):

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

    train_pairs = [(k, get_file_text(v)) for k, v in list(zip(train_im, train_txt))]
    test_pairs = [(k, get_file_text(v)) for k, v in list(zip(test_im, test_txt))]

    return (train_pairs, test_pairs)
