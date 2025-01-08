import dataclasses
from random import randint
from dataclasses import dataclass
from transformers import pipeline, AutoTokenizer, AdamW, AutoModelForSequenceClassification


@dataclass
class PlmConfig:
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
    hf_plm_name: str = "almanach/camembertv2-base"
    max_input_length: int = 512
    ckpt_path: str = "./ckpt"


class DataProcessor:

    def __init__(self,cfg: PlmConfig):
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.hf_plm_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(cfg.hf_plm_name)
        self.fillmask = pipeline("fill-mask", model=cfg.hf_plm_name)
        self.mask_token = self.fillmask.tokenizer.mask_token

    def augment_data(self,examples):
        outputs = []
        for sentence in examples["Avis"]:
            words = sentence.split(' ')
            K = randint(1, len(words) - 1)
            masked_sentence = " ".join(words[:K] + [self.mask_token] + words[K + 1:])
            predictions = self.fillmask(masked_sentence)
            augmented_sequences = [predictions[i]["sequence"] for i in range(2)]
            outputs += [sentence] + augmented_sequences
        return {"data": outputs}


##class PLMClassifier:



