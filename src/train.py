import os

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from torch.nn import CTCLoss
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from pathlib import Path
from torch import optim

from config import train_config as config
from dataset import SynthDataset, collate_fn, splitter
from model import CRNN


torch.backends.cudnn.enabled = False

class SaveFianlModel(Callback):

    def __init__(self, pth):
        super().__init__()
        self.pth = pth
        self.best_loss = 1000000000

    def on_validation_epoch_end(self, trainer, pl_module):

        loss = trainer.callback_metrics["loss_val"]
        if loss < self.best_loss:
            self.best_loss = loss
            sample = torch.rand(1, 1, 32, 96).cuda()
            scripted_model = torch.jit.trace(pl_module.model.eval(), sample)
            scripted_model.save(f'{self.pth}_{loss:.2f}vl_CRNN_custom_tuned_augs.pth')


class PLWrap(pl.LightningModule):
    """
    Lightning model wrap class
    """

    def __init__(self, model, lr):
        super().__init__()
        self.lr = lr
        self.model = model
        self.save_hyperparameters()
        self.acc = Accuracy()
        self.loss_fn = CTCLoss(reduction="sum", zero_infinity=True)

    def _reconstruct(self, labels, blank=0):
        new_labels = []
        # merge same labels
        previous = None
        for l in labels:
            if l != previous:
                new_labels.append(l)
                previous = l
        # delete blank
        new_labels = [l for l in new_labels if l != blank]

        return new_labels

    def _greedy_decode(self, emission_log_prob, blank=0):
        labels = np.argmax(emission_log_prob, axis=-1)
        labels = self._reconstruct(labels, blank=blank)
        return labels

    def ctc_decode(self, log_probs, label2char=None, blank=0):

        emission_log_probs = np.transpose(log_probs.cpu().numpy(), (1, 0, 2))
        # size of emission_log_probs: (batch, length, class)

        decoded_list = []
        for emission_log_prob in emission_log_probs:
            decoded = self._greedy_decode(emission_log_prob, blank=blank)
            if label2char:
                decoded = [label2char[l] for l in decoded]
            decoded_list.append(decoded)
        return decoded_list

    def training_step(self, batch, batch_idx):

        images, targets, target_lengths = batch

        logits = self.model(images)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        batch_size = images.size(0)
        input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
        target_lengths = torch.flatten(target_lengths)

        loss = self.loss_fn(log_probs, targets, input_lengths, target_lengths)

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 5
        )  # gradient clipping with 5

        preds = self.ctc_decode(log_probs.detach())
        target_lengths = target_lengths.cpu().numpy().tolist()
        reals = targets.cpu().numpy().tolist()
        self._log(
            loss.item() / batch_size,
            preds,
            target_lengths,
            reals,
            batch_size=batch_size,
            desc="train",
        )

        return {
            "loss": loss / batch_size,
            "preds": preds,
            "target_lens": target_lengths,
            "reals": reals,
        }

    def _log(self, loss, preds, target_lengths, reals, batch_size, desc=""):

        tot_count = batch_size
        tot_correct = 0
        wrong_cases = []
        target_length_counter = 0
        for pred, target_length in zip(preds, target_lengths):
            real = reals[target_length_counter : target_length_counter + target_length]
            target_length_counter += target_length
            if pred == real:
                tot_correct += 1
            else:
                wrong_cases.append((real, pred))

        self.log(
            "loss_" + desc, loss, on_epoch=True, on_step=False, batch_size=batch_size
        )
        self.log(
            "acc_" + desc,
            tot_correct / (tot_count + 0.000001),
            on_epoch=True,
            on_step=False,
            batch_size=batch_size,
        )

    def validation_step(self, batch, batch_idx):
        images, targets, target_lengths = batch

        logits = self.model(images)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        batch_size = images.size(0)
        input_lengths = torch.LongTensor([logits.size(0)] * batch_size)

        loss = self.loss_fn(log_probs, targets, input_lengths, target_lengths)

        preds = self.ctc_decode(log_probs)
        target_lengths = target_lengths.cpu().numpy().tolist()
        reals = targets.cpu().numpy().tolist()

        self._log(
            loss.item() / batch_size, preds, target_lengths, reals, batch_size, "val"
        )

        return {
            "losses": loss / batch_size,
            "preds": preds,
            "target_lens": target_lengths,
            "reals": reals,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), 
                                      weight_decay=0.0001, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                          milestones=[5], gamma=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


def main():
    epochs = config["epochs"]
    train_batch_size = config["train_batch_size"]
    eval_batch_size = config["eval_batch_size"]
    lr = config['lr']

    cpu_workers = config["cpu_workers"]

    img_width = config["img_width"]
    img_height = config["img_height"]
    data_dir = config["data_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    torch.backends.cudnn.benchmark = True

    (train_pairs, test_pairs) = splitter(data_dir, 0.2)

    RUS_LETTERS = "АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя"
    NUMBERS = "0123456789"
    PUNCTUATION_MARKS = ' .,?:;—!<>-«»()[]*"'
    voc = RUS_LETTERS + NUMBERS + PUNCTUATION_MARKS

    train_dataset = SynthDataset(
        pairs=train_pairs, train_mode=True, img_height=img_height, img_width=img_width, voc=voc
    )
    valid_dataset = SynthDataset(
        pairs=test_pairs, train_mode=False, img_height=img_height, img_width=img_width, voc=voc
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=cpu_workers,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=eval_batch_size,
        pin_memory=True,
        num_workers=cpu_workers,
        collate_fn=collate_fn,
    )

    num_class = len(voc) + 1

    crnn = CRNN(
        1,
        img_height,
        img_width,
        num_class,
        map_to_seq_hidden=config["map_to_seq_hidden"],
        rnn_hidden=config["rnn_hidden"],
        leaky_relu=config["leaky_relu"],
    )
    
    jit_lodel = torch.jit.load('/home/mangaboba/environ/passport_ocr/_0.33vl_CRNN128_sctipted.pth')
    model_dict = jit_lodel.state_dict()
    crnn.load_state_dict(model_dict)
    del jit_lodel
    del model_dict

    model = PLWrap(crnn, lr)

    wandb.login(key=os.environ['WANDB_KEY'].strip())
    wandb.init()

    wandb_logger = WandbLogger(
        project="ocr_task",
        entity="mangaboba",
        log_model=False,
    )

    val_checkpoint = ModelCheckpoint(
        filename="crnn" + "_valloss-{epoch}-{step}-{val_loss}",
        monitor="loss_val",
        mode="min",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        default_root_dir=config["checkpoints_dir"],
        gpus=1,
        max_epochs=epochs,
        max_steps=-1,
        logger=[wandb_logger],
        callbacks=[val_checkpoint, SaveFianlModel(config["jit_pth"])],
        strategy="ddp",
    )
    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    main()
