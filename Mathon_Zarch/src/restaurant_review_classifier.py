import glob
import os
from dataclasses import dataclass

import lightning as L
import pandas as pd
import pyrallis
import torch
from datasets import Dataset
from lightning.pytorch.callbacks import EarlyStopping
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from transformers import AutoConfig, AutoTokenizer, AutoModel, get_scheduler


@dataclass
class GlobalConfig:
    device: int = 0
    dropout: float = 0.2
    batch_size: int = 32  # prend plus 12Go de VRAM sur ma machine
    # l'entraînement repart de la dernière sauvegarde
    n_runs: int = 5
    # n_train is the number of samples on which to run the eval. n_trian=-1 means eval on all test data,
    n_train: int = -1
    # n_test is the number of samples on which to run the eval. n_test=-1 means eval on all test data,
    n_test: int = -1
    lr: float = 1e-5
    max_epochs: int = n_runs * 5  # le training est stoppé après 3 epochs sans amélioration donc on peut se le permettre
    num_workers: int = 0
    hf_plm_name: str = "almanach/camembertv2-base"  # French BERT model
    max_input_length: int = 512
    dataset_train_path = "../data/ftdataset_train.tsv"
    dataset_val_path = "../data/ftdataset_val.tsv"
    dataset_test_path = "../data/ftdataset_test.tsv"
    logging_root_dir: str = "."
    ckpt_root_dir: str = "./ckpt"


class RestaurantDataProcessor:
    def __init__(self, cfg: GlobalConfig, tokenizer, aspect_labels: dict):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.aspect_labels = aspect_labels
        self.label2id = {aspect: {label: idx for idx, label in enumerate(labels)}
                         for aspect, labels in aspect_labels.items()}
        self.id2label = {aspect: {idx: label for idx, label in enumerate(labels)}
                         for aspect, labels in aspect_labels.items()}

    def collate_fn(self, samples: list[dict]) -> dict:
        input_texts = [sample['Avis'] for sample in samples]
        encoded_input = self.tokenizer(
            input_texts,
            add_special_tokens=True,
            padding='longest',
            truncation=True,
            max_length=self.cfg.max_input_length,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Encode labels for each aspect
        encoded_labels = {}
        for aspect in self.aspect_labels.keys():
            if aspect in samples[0]:
                labels = torch.zeros(size=(len(samples),), dtype=torch.long)
                for idx, sample in enumerate(samples):
                    if sample[aspect] in self.label2id[aspect]:
                        labels[idx] = self.label2id[aspect][sample[aspect]]
                encoded_labels[aspect] = labels

        batch = {
            'input_ids': encoded_input.input_ids,
            'attention_mask': encoded_input.attention_mask,
            **encoded_labels
        }
        return batch

    def decode_labels(self, predictions: dict) -> dict:
        decoded = {}
        for aspect, pred_ids in predictions.items():
            decoded[aspect] = [self.id2label[aspect][id.item()] for id in pred_ids]
        return decoded


class MultiAspectClassifier(nn.Module):
    def __init__(self, input_dim: int, aspect_dims: dict, dropout: float):
        super().__init__()
        self.aspect_classifiers = nn.ModuleDict({
            aspect: nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim * 2, n_labels)
            )
            for aspect, n_labels in aspect_dims.items()
        })

    def forward(self, X):
        return {aspect: classifier(X) for aspect, classifier in self.aspect_classifiers.items()}


class RestaurantReviewClassifier(L.LightningModule):
    def __init__(self, cfg: GlobalConfig, aspect_labels: dict):
        super().__init__()
        self.cfg = cfg
        self.aspect_labels = aspect_labels
        torch.set_float32_matmul_precision('medium')
        # Load pretrained language model
        self.config = AutoConfig.from_pretrained(cfg.hf_plm_name)
        self.lmtokenizer = AutoTokenizer.from_pretrained(cfg.hf_plm_name)
        self.lm = AutoModel.from_pretrained(cfg.hf_plm_name)
        self.lm.train()

        # Multi-aspect classifier
        aspect_dims = {aspect: len(labels) for aspect, labels in aspect_labels.items()}
        self.classifier = MultiAspectClassifier(
            input_dim=self.config.hidden_size,
            aspect_dims=aspect_dims,
            dropout=cfg.dropout
        )

        # Data processor
        self.data_processor = RestaurantDataProcessor(cfg, self.lmtokenizer, aspect_labels)

        # Loss function and metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.metrics = nn.ModuleDict({
            f"{split}_{aspect}_accuracy": Accuracy(task="multiclass", num_classes=len(labels))
            for split in ['val', 'test']
            for aspect, labels in aspect_labels.items()
        })

    def forward(self, batch):
        # Get embeddings
        outputs = self.lm(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])[0]
        # Pool embeddings (mean pooling)
        mask = batch['attention_mask'].unsqueeze(-1).expand(outputs.size())
        sum_embeddings = torch.sum(outputs * mask, 1)
        sum_mask = mask.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        pooled_output = sum_embeddings / sum_mask
        # Get predictions for each aspect
        return self.classifier(pooled_output)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)
        scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=0, num_training_steps=10)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        predictions = self(batch)
        loss = 0
        for aspect in self.aspect_labels.keys():
            if aspect in batch:
                aspect_loss = self.loss_fn(predictions[aspect], batch[aspect])
                loss += aspect_loss
                self.log(f"train_{aspect}_loss", aspect_loss)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        predictions = self(batch)
        loss = 0
        for aspect in self.aspect_labels.keys():
            if aspect in batch:
                aspect_loss = self.loss_fn(predictions[aspect], batch[aspect])
                loss += aspect_loss
                pred_labels = torch.argmax(predictions[aspect], dim=-1)
                self.metrics[f"val_{aspect}_accuracy"](pred_labels, batch[aspect])
                self.log(f"val_{aspect}_loss", aspect_loss)
                self.log(f"val_{aspect}_accuracy", self.metrics[f"val_{aspect}_accuracy"])
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        predictions = self(batch)
        loss = 0
        for aspect in self.aspect_labels.keys():
            if aspect in batch:
                aspect_loss = self.loss_fn(predictions[aspect], batch[aspect])
                loss += aspect_loss
                pred_labels = torch.argmax(predictions[aspect], dim=-1)
                self.metrics[f"test_{aspect}_accuracy"](pred_labels, batch[aspect])
                self.log(f"test_{aspect}_loss", aspect_loss)
                self.log(f"test_{aspect}_accuracy", self.metrics[f"test_{aspect}_accuracy"])
        self.log("test_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        predictions = self(batch)
        return {aspect: torch.argmax(pred, dim=-1)
                for aspect, pred in predictions.items()}


'''
def augment_data(examples, fill_mask, mask_token):
    outputs = []
    for sentence in examples["Avis"]:
        words = sentence.split(' ')
        K = randint(1, len(words) - 1)
        masked_sentence = " ".join(words[:K] + [mask_token] + words[K + 1:])
        predictions = fill_mask(masked_sentence)
        augmented_sequences = [predictions[i]["sequence"] for i in range(1)]
        outputs += augmented_sequences

    return {"Avis": outputs}
'''


def prepare_data(cfg, df_train=None, df_val=None, df_test=None):
    # Read CSV
    # Ce n'est pas propre, mais ça permet de la faire tourner via runproject
    if df_train is None:
        df_train = pd.read_csv(cfg.dataset_train_path, sep=' *\t *', encoding='utf-8', engine='python')
        df_train = Dataset.from_pandas(df_train)
    if df_val is None:
        df_val = pd.read_csv(cfg.dataset_val_path, sep=' *\t *', encoding='utf-8', engine='python')
        df_val = Dataset.from_pandas(df_val)
    if df_test is None:
        df_test = pd.read_csv(cfg.dataset_test_path, sep=' *\t *', encoding='utf-8', engine='python')

    # Get unique labels for each aspect
    aspect_labels = {
        'Prix': sorted(df_test['Prix'].unique().tolist()),
        'Cuisine': sorted(df_test['Cuisine'].unique().tolist()),
        'Service': sorted(df_test['Service'].unique().tolist()),
        'Ambiance': sorted(df_test['Ambiance'].unique().tolist())
    }

    # Tout ce bazard pour augmenter le dataset
    '''
    fill_mask = pipeline("fill-mask", model="almanach/camembertv2-base", use_fast=True, torch_dtype=torch.bfloat16)
    mask_token = fill_mask.tokenizer.mask_token
    df_train = pd.DataFrame(df_train)
    sub_length = math.ceil(len(df_train) * 0.1)
    K = randint(1, len(df_train) - sub_length)
    df_train_subset = Dataset.from_pandas(df_train[K:K + sub_length - 1])
    df_train_subset.map(augment_data, batched=True, remove_columns=df_train_subset.column_names, batch_size=32,
                        fn_kwargs={"fill_mask": fill_mask, "mask_token": mask_token})
    df_train_subset = pd.DataFrame(df_train_subset)
    df_train = pd.concat([df_train, df_train_subset], ignore_index=True)
    df_train = Dataset.from_pandas(df_train)
    '''
    # Convert to Dataset
    df_test = Dataset.from_pandas(df_test)

    return {
        'train': df_train,
        'validation': df_val,
        'test': df_test
    }, aspect_labels


def train_model(cfg: GlobalConfig, df_train=None, df_val=None):
    # Prepare data
    datasets, aspect_labels = prepare_data(cfg, df_train, df_val)

    # Create model
    model = RestaurantReviewClassifier(cfg, aspect_labels)

    last_checkpoint = None
    last_checkpoint = find_latest_checkpoint(cfg.logging_root_dir)

    # Create dataloaders
    dataloaders = {
        split: DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=(split == 'train'),
            collate_fn=model.data_processor.collate_fn,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )
        for split, dataset in datasets.items()
    }

    # Setup training (Il faut l'installer dommage)
    # logger = TensorBoardLogger(save_dir=cfg.logging_root_dir)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=3,
        verbose=True
    )

    # Create trainer
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator='gpu',
        devices=[cfg.device],
        callbacks=[early_stopping],
        precision='bf16-mixed'
    )

    # Train
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}")
        # Resume training from checkpoint
        trainer.fit(
            model=model,
            train_dataloaders=dataloaders['train'],
            val_dataloaders=dataloaders['validation'],
            ckpt_path=last_checkpoint
        )
    else:
        print(f"No checkpoint found. Starting training from scratch.")
        # Start training from scratch
        trainer.fit(
            model=model,
            train_dataloaders=dataloaders['train'],
            val_dataloaders=dataloaders['validation']
        )

    return model, trainer, dataloaders


def find_latest_checkpoint(logging_dir: str) -> str:
    """Find the latest checkpoint across all versions."""
    # Find all version directories
    version_dirs = glob.glob(os.path.join(logging_dir, "lightning_logs", "version_*"))
    if not version_dirs:
        return None

    # Find all checkpoints across all versions
    all_checkpoints = []
    for version_dir in version_dirs:
        checkpoint_dir = os.path.join(version_dir, "checkpoints")
        if os.path.exists(checkpoint_dir):
            checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
            all_checkpoints.extend(checkpoints)

    if not all_checkpoints:
        return None

    # Return the most recent checkpoint based on modification time
    latest_checkpoint = max(all_checkpoints, key=os.path.getmtime)

    # Get version number from path
    version = latest_checkpoint.split("version_")[1].split("/")[0]
    print(f"Found latest checkpoint in version_{version}: {latest_checkpoint}")

    return latest_checkpoint


if __name__ == "__main__":
    cfg = pyrallis.parse(config_class=GlobalConfig)
    # Train model
    model, trainer, dataloaders = train_model(cfg)

    # Test
    trainer.test(model, dataloaders=dataloaders['test'])

    # Example prediction
    test_reviews = [
        "Le service était excellent, la nourriture délicieuse et l'ambiance très agréable. Les prix sont raisonnables.",
        "Déçu par la qualité de la cuisine. Le service était correct mais les prix trop élevés."
    ]

    # Create test samples
    test_samples = [{'Avis': review} for review in test_reviews]
    test_loader = DataLoader(
        test_samples,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=model.data_processor.collate_fn
    )

    # Get predictions
    predictions = trainer.predict(model, dataloaders=test_loader)
    # Combine predictions from all batches
    combined_predictions = {
        aspect: torch.cat([batch[aspect] for batch in predictions])
        for aspect in model.aspect_labels.keys()
    }
    # Decode predictions
    decoded_predictions = model.data_processor.decode_labels(combined_predictions)

    # Print results
    for i, review in enumerate(test_reviews):
        print(f"\nReview: {review}")
        for aspect in model.aspect_labels.keys():
            print(f"{aspect}: {decoded_predictions[aspect][i]}")
