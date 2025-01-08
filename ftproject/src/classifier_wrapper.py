from tqdm import tqdm


from config import Config

from llm_classifier import LLMClassifier

class ClassifierWrapper:

    # METTRE LA BONNE VALEUR ci-dessous en fonction de la méthode utilisée
    METHOD: str = 'LLM'  # or 'PLMFT' (for Pretrained Language Model Fine-Tuning)



    #############################################################################################
    # NE PAS MODIFIER LA SIGNATURE DE CETTE FONCTION, Vous pouvez modifier son contenu si besoin
    #############################################################################################
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.classifier = LLMClassifier(cfg)



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
        # Comme ici on utilise un llm avce ollama, on procèdera en traitant les textes d'avis un à un
        # mais si on utilise un PLMFT, il vaut mieux traiter les avis par batch pour que ce soit plus
        # rapide
        for text in tqdm(texts):
            opinions = self.classifier.predict(text)
            all_opinions.append(opinions)
        return all_opinions




