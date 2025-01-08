from dataclasses import dataclass, field
from pprint import pprint
from typing import Any

import pyrallis
from datasets import Dataset, load_dataset
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.preprocessing import LabelBinarizer
import torch
import torch.nn.functional as F
from torch import nn
import lightning as L
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from transformers import AutoConfig, AutoTokenizer, AutoModel, PreTrainedTokenizerBase, DataCollatorWithPadding


############################### Global configuration
@dataclass
class GlobalConfig:
    # list of devices on which model is run
    device: int = 0
    # dropout
    dropout: int = 0.2
    # batch size
    batch_size: int = 16
    # learning rate
    lr: float = 1e-5
    # max number of epochs
    max_epochs: int = 50
    # number of workers for the data loaders
    num_workers: int = 0
    # Name of the HF pretrained MLM to use as an encoder
    hf_plm_name: str = "almanach/camembert-base" # "prajjwal1/bert-tiny"
    max_input_length: int = 512
    # dataset name
    # hf_dataset_name: str = "fancyzhx/ag_news"
    hf_dataset_name: str = "mteb/sib200"
    hf_dataset_subset: str = "fra_Latn"
    # Logging root dir: where to save the logger outputs
    logging_root_dir: str = "./logging"
    # Where to save the checkpoints
    ckpt_root_dir: str = "./ckpt"



#################################### Utilities

def pool_mean_embeddings(token_embeddings, attention_mask):
    # we reshape the mask to add a 3rd dimension (corresponding to the embedding dim) and
    # we expand it by duplicating its binary values along that dimension. We obtain
    # a (N, seq_len, emb_dim) tensor (similar to the token embeddings shape)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # we multiply the token embeddings and the expanded mask (element-wise) to set
    # to zero all the embedding values corresponding to padding positions; and then
    # we sum the modified embeddings along dim 1 (time dimension of the token sequences).
    # padding positions are excluded from this sum because their values were set to
    # zero by the element-wise multiplication with the expanded mask.
    # resulting shape = (N, emb_dim)
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
    # we also sum the expanded mask values along the time dimension: we obtain counts
    # of non-padding tokens for each sequence in the batch, shape=(N, emb_dim)
    sum_mask = input_mask_expanded.sum(dim=1)
    # we will divide the sum_embeddings by the sum_mask (aka. counts of non padding tokens)
    # to get the mean embeddings. In order to avoid division by zero (which might happen,
    # in theory, when there is an empty token sequence that contains only padding
    # tokens), we first clamp the sum_mask with a very small non-zero value
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    # divide the sum_embeddings by the sum_mask to get the mean embeddings
    mean_embeddings = sum_embeddings / sum_mask
    # shape = (N, emb_dim)
    return mean_embeddings


##################################### DATA PROCESSING


class DataProcessor:

    def __init__(self, cfg: GlobalConfig, tokenizer, labels: list[str]):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.labels = labels
        self.label2id = {label:idx for idx, label in enumerate(labels)}
        self.id2label = {idx:label for idx, label in enumerate(labels)}

    def collate_fn(self, samples: list[dict]) -> dict:
        # inputs
        input_texts = [sample['text'] for sample in samples]
        encoded_input = self.tokenizer(input_texts, add_special_tokens=True, padding='longest', truncation=True,
                                       max_length=self.cfg.max_input_length, return_attention_mask=True,
                                       return_tensors='pt', return_offsets_mapping=False, return_token_type_ids=False,
                                       verbose=False, )
        N = len(samples)
        if 'category' in samples[0]:
            # samples have the tru labels: encode them too
            encoded_labels = torch.zeros(size=(N,), dtype=torch.long)
            for idx, sample in enumerate(samples):
                encoded_labels[idx] = self.label2id[sample['category']]
        else:
            encoded_labels = None
        # build batch
        batch = {
            'input_ids': encoded_input.input_ids,
            'attention_mask': encoded_input.attention_mask,
            'label_ids': encoded_labels,
        }
        return batch

    def decode_labels(self, label_ids: list[int]) -> list[str]:
        return [self.id2label[id] for id in label_ids]


###################################### MODEL



# First, a classifier component (a simple PyTorch NN module)
class ClassifierComponent(nn.Module):

    def __init__(self, input_dim: int, n_labels: int, dropout: float):
        super().__init__()
        k = 2
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * k),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * k, n_labels),
        )

    def forward(self, X):
        """
        :param X:   B x d
        :return:
        """
        return self.ffn(X)  # B x C   (C = number of classes = n_labels)



# Main module (a Lightnning module)
class MLMBasedClassifier(L.LightningModule):

    def __init__(self, cfg: GlobalConfig, labels: list[str]):
        super(MLMBasedClassifier, self).__init__()
        self.cfg = cfg
        # text encoder: a pretrained language model (PLM)
        self.config = AutoConfig.from_pretrained(cfg.hf_plm_name)
        self.lmtokenizer = AutoTokenizer.from_pretrained(cfg.hf_plm_name)
        self.lm = AutoModel.from_pretrained(cfg.hf_plm_name, output_attentions=False, add_pooling_layer=False)
        # When loading the pretrained LM, it is set on eval mode. We need to put it in train mode if
        # we want to fine-tune its parameters:
        self.lm.train()
        # classifier component:
        num_classes = len(labels)
        self.classifier_component = ClassifierComponent(input_dim=self.config.hidden_size, n_labels=num_classes, dropout=cfg.dropout)
        # Data collator (needed for building batches)
        self.data_processor = DataProcessor(cfg, self.lmtokenizer, labels)
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        # Eval metric to use: accuracy
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, batch):
        out = self.lm(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])[0]
        # The following is an imperfect average pooling (because padding tokens are included in the average)
        # out = out.mean(dim=1)
        # The following averaging is better because it discards the padding tokens when computing the mean
        out = pool_mean_embeddings(out, batch['attention_mask'])
        # apply classification layers
        out = self.classifier_component(out)
        return out

    # This is the function where the optimizer settings are defined
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        return optimizer

    # the following func is called for every batch when trainer.fit() is called (with a train_dataloader)
    def training_step(self, batch, batch_idx):
        # training_step is called for every training step (i.e. every batch)
        # Apply the model to this input batch
        y_hat = self(batch)
        # batch['labels'] contains the ground-truth labels for this batch: compute the loss
        loss = self.loss_fn(y_hat, batch['label_ids'])
        # Log the train loss at every epoch
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # must return the loss
        return loss

    # the following func is called for every batch when trainer.fit() is called with a validation dataloader
    def validation_step(self, batch, batch_ix):
        # Validation_step is called after every train step (i.e. for every batch)
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, batch['label_ids'])
        # compute the accuracy for this batch using the torchmetrics Accuracy object
        y_hat = torch.argmax(y_hat, dim=-1)
        self.val_accuracy(y_hat, batch['label_ids'])
        # Log the validation loss and accuracy
        self.log_dict(
            dictionary={'val_loss': loss.item(), 'val_acc': self.val_accuracy},
            on_step=False, on_epoch=True, prog_bar=True,
            sync_dist=True,
        )
        return loss

    # the following func is called for every batch when trainer.test() is called with a dataloader
    def test_step(self, batch, batch_idx):
        # this is the test loop
        y_hat = self(batch)
        # compute the accuracy for this batch using the torchmetrics Accuracy object
        y_hat = torch.argmax(y_hat, dim=-1)
        self.test_accuracy(y_hat, batch['label_ids'])
        self.log("test_acc", self.test_accuracy, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        y_hat = self(batch)   # N,C   (where C=number of classes)
        # get argmax
        y_hat = torch.argmax(y_hat, dim=-1)  # N,
        pred_labels = self.data_processor.decode_labels(y_hat.tolist())
        return pred_labels


    def predict(self, texts: list[str]) -> list[str]:
        samples = [{'text': text} for text in texts]
        # # partition the list of samples into chunks of size = cfg.batch_size
        # chunks = [samples[i:i + self.cfg.batch_size] for i in range(0, len(samples), self.cfg.batch_size)]
        # all_preds = []
        # for chunk in chunks:
        #     batch = self.data_processor.collate_fn(chunk)
        #     preds = self.predict_step(batch)
        #     all_preds.extend(preds)
        data_loader = DataLoader(samples, shuffle=False, batch_size=cfg.batch_size, collate_fn=model.data_processor.collate_fn, num_workers=cfg.num_workers)
        trainer = L.Trainer(accelerator='gpu', devices=[cfg.device], strategy='auto')
        # predict
        preds = trainer.predict(model, dataloaders=data_loader)  # a list of list of predictions
        # flatten predictions
        preds = [pred for pred_list in preds for pred in pred_list]
        return preds


def create_and_train_model(cfg: GlobalConfig) -> MLMBasedClassifier:
    # Prepare data splits (no dev split in the original dataset, so we'll use a small part of the train as a dev)
    ds_train = load_dataset(cfg.hf_dataset_name, cfg.hf_dataset_subset, split='train')   # 24000
    ds_dev = load_dataset(cfg.hf_dataset_name, cfg.hf_dataset_subset, split='validation')
    # get set of labels
    labels = list(set(ds_train['category']))
    model = MLMBasedClassifier(cfg, labels)
    # dataloaders
    dl_train = DataLoader(ds_train, shuffle=True, batch_size=cfg.batch_size, collate_fn=model.data_processor.collate_fn, num_workers=cfg.num_workers)
    dl_dev = DataLoader(ds_dev, shuffle=False, batch_size=cfg.batch_size, collate_fn=model.data_processor.collate_fn, num_workers=cfg.num_workers)
    # Create a logger
    logger = TensorBoardLogger(save_dir=cfg.logging_root_dir)
    # Create an early stopping callback
    earlystop_cbk = EarlyStopping(
        monitor='val_loss', mode='min', min_delta=0.00, patience=5, verbose=True,
    )
    # create a Lighting trainer
    trainer = L.Trainer(
        # limit_train_batches=5, limit_val_batches=5,  # this is for debugging only
        max_epochs=cfg.max_epochs,
        accelerator='gpu', devices=[cfg.device],
        # The following is only if multi-device AND forward() may have unused model parameters
        # strategy='ddp_find_unused_parameters_true',
        strategy='auto',
        callbacks=[earlystop_cbk],
        logger=logger
    )
    # Training
    trainer.fit(model=model, train_dataloaders=dl_train, val_dataloaders=dl_dev)
    return model


def test_model(model: MLMBasedClassifier):
    # Testing
    print("Testing...")
    ds_test = load_dataset(cfg.hf_dataset_name, cfg.hf_dataset_subset, split='test')
    dl_test = DataLoader(ds_test, shuffle=False, batch_size=cfg.batch_size, collate_fn=model.data_processor.collate_fn, num_workers=cfg.num_workers)
    trainer = L.Trainer(accelerator='gpu', devices=[cfg.device], strategy='auto')
    # Test
    trainer.test(model, dataloaders=dl_test)


if __name__ == "__main__":
    # Prepare the global configuration
    cfg = pyrallis.parse(config_class=GlobalConfig)
    pprint(vars(cfg), sort_dicts=False, compact=True)
    # Create Lightning model using the config and train it
    model = create_and_train_model(cfg)
    # Testing
    test_model(model)
    # prediction
    texts = [
        "Le Vatican compte environ 800 habitants. C'est le plus petit pays indépendant au monde, et le moins peuplé.",
        "L'expérience Hershey et Chase a été l'une des principales indications que l'ADN était un matériel génétique.",
        "On peut randonner dans les montagnes autour de Grenoble.",
    ]
    labels = model.predict(texts)
    print(labels)



