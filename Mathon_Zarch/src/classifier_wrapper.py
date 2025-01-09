import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from Mathon_Zarch.src.restaurant_review_classifier import GlobalConfig, train_model
from llm_classifier import LLMClassifier


class ClassifierWrapper:

    # METTRE LA BONNE VALEUR ci-dessous en fonction de la méthode utilisée
    METHOD: str = 'PLMFT'  # or 'LLM' (for Pretrained Language Model Fine-Tuning)



    #############################################################################################
    # NE PAS MODIFIER LA SIGNATURE DE CETTE FONCTION, Vous pouvez modifier son contenu si besoin
    #############################################################################################
    def __init__(self, cfg: Config):
        self.model = None
        self.trainer = None
        self.dataloaders = None
        self.cfg = cfg
        if self.METHOD == 'LLM':
            self.classifier = LLMClassifier(cfg)
        elif self.METHOD == 'PLMFT':
            self.global_cfg = GlobalConfig()




    #############################################################################################
    # NE PAS MODIFIER LA SIGNATURE DE CETTE FONCTION, Vous pouvez modifier son contenu si besoin
    #############################################################################################
    def train(self, train_data: list[dict], val_data: list[dict], device: int) -> None:
        """
        :param train_data:
        :param val_data:
        :param device: un nombre qui identifie le numéro de la gpu sur laquelle le traitement doit se faire
        -1 veut deire que le device est la cpu, et un nombre entier >= 0 indiquera le numéro de la gpu
        :return:
        """
        # Mettre tout ce qui est nécessaire pour entrainer le modèle ici, sauf si methode=LLM en zéro-shot
        # auquel cas pas d'entrainement du tout
        if self.METHOD == 'PLMFT':
            self.model, self.trainer, self.dataloaders = train_model(self.global_cfg,train_data,val_data)
        pass



    #############################################################################################
    # NE PAS MODIFIER LA SIGNATURE DE CETTE FONCTION, Vous pouvez modifier son contenu si besoin
    #############################################################################################
    def predict(self, texts: list[str], device: int) -> list[dict]:
        """
        :param texts:
        :param device: device: un nombre qui identifie le numéro de la gpu sur laquelle le traitement doit se faire
        -1 veut deire que le device est la cpu, et un nombre entier >= 0 indiquera le numéro de la gpu à utiliser
        :return:
        """
        all_opinions = []
        if self.METHOD == 'LLM':
            for text in tqdm(texts):
                opinions = self.classifier.predict(text)
                all_opinions.append(opinions)
        elif self.METHOD == 'PLMFT':
            # Create test samples
            test_samples = [{'Avis': review} for review in texts]
            test_loader = DataLoader(
                test_samples,
                batch_size=self.global_cfg.batch_size,
                shuffle=False,
                collate_fn=self.model.data_processor.collate_fn
            )
            predictions = self.trainer.predict(self.model, dataloaders=test_loader)
            combined_predictions = {
                aspect: torch.cat([batch[aspect] for batch in predictions])
                for aspect in self.model.aspect_labels.keys()
            }
            decoded_predictions = self.model.data_processor.decode_labels(combined_predictions)
            for i, text in enumerate(texts):
                opinions = {aspect: decoded_predictions[aspect][i] for aspect in self.model.aspect_labels.keys()}
                all_opinions.append(opinions)
        return all_opinions




