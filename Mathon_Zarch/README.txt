Auteurs: Mathon Tristan et Zarch Vassili

## Description du modèle

Le modèle fait 111M + 4.7M de paramètres.

Le classifieur utilisé pour la classification des avis de restaurants est basé sur un modèle de langage pré-entraîné (PLM)
appelé camembertv2-base, un modèle BERT en français.
Ce modèle est enrichi d'une couche de classification multi-aspects pour prédire les aspects suivants :
Prix, Cuisine, Service et Ambiance.
La représentation des textes est effectuée via le tokenizer du modèle, qui convertit les avis en séquences de tokens
adaptées à l'entrée du modèle BERT.
L'architecture du classifieur comprend un module PyTorch MultiAspectClassifier qui contient des classificateurs pour chaque aspect.
Chaque classificateur est une séquence de couches linéaires et de fonctions d'activation.

                nn.Linear(input_dim, input_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim * 2, n_labels)

Le modèle est entraîné en utilisant PyTorch Lightning avec un optimiseur AdamW et un scheduler de taux d'apprentissage.
Un callback d'arrêt anticipé est utilisé pour arrêter l'entraînement si le modèle ne s'améliore pas pendant 3 epochs consécutives.

Les résultats obtenus (avec 25 epochs et 3500 steps et sans data augmentation) sont les suivants :
    ALL RUNS ACC: [83.26, 83.45, 83.45, 83.45, 83.45]
    AVG MACRO ACC: 83.41
    TOTAL EXEC TIME: 1672.3 (soit presque 28 mins)

Les résultats obtenus (avec 25 epochs et 3850 steps et avec data augmentation de 10%) sont les suivants :
   ALL RUNS ACC: [82.65, 82.65, 82.65, 82.65, 82.65]
   AVG MACRO ACC: 82.65
   TOTAL EXEC TIME: 277.5

Les temps affichés sur ce test ne correspondent pas aux temps d'entraînement, car le classifier à réutiliser
le dernier checkpoint. La data augmentation a été implémentée pour essayer d'augmenter la précision du modèle, mais cela n'a pas eu d'effet significatif.
En plus d'augmenter le temps d'entraînement, la consommation de VRAM sur les différentes runs a fortement augmenté, ce qui a nécessité de relancer le training pour remettre la VRAM à zéro.
La data augmentation n'a pas été laissé dans la version finale pour des raisons de simplicité et de vitesse d'entraînement.

Recharger le modèle permet d'économiser du temps de calcul, mais ne permet pas d'entraîner le modèle une fois le nombre d'epochs atteint.
Pour déactiver le rechargement du modèle, il suffit de commenter la ligne 263 de 'restaurant_review_classifier.py'.
Le nombre d'epochs dépend du nombre de runs, et est fixé à cinq par run (25 epochs au total est proche de la limite en terme d'entraînement).

Le temps d'entraînement est aux alentours d'une minute par epoch environ (soit ~25 mins d'entraînement au total).
Le trainer utilise comme précision "16 bit float mixed" pour accélérer l'entraînement et réduire la mémoire (au lieu du 32 bit full précision).



La totalité des fichiers '.ipynb' sont des fichiers "bac à sable" pour tester des fonctionnalités, des idées ou des modifications.

Les Hyper-paramètres :
class GlobalConfig:
    device: int = 0
    dropout: float = 0.2
    batch_size: int = 32 # prend plus 12Go de VRAM sur ma machine (au bout d'un certain nombre d'epochs, il vaut mieux relancer pour remettre la VRAM à zéro)
    # l'entraînement repart de la dernière sauvegarde
    n_runs: int = 5
    # n_train is the number of samples on which to run the eval. n_trian=-1 means eval on all test data,
    n_train: int = -1
    # n_test is the number of samples on which to run the eval. n_test=-1 means eval on all test data,
    n_test: int = -1
    lr: float = 1e-5
    max_epochs: int = n_runs * 5 # le training est stoppé après 3 epochs sans amélioration donc on peut se le permettre
    num_workers: int = 0
    hf_plm_name: str = "almanach/camembertv2-base"  # French BERT model
    max_input_length: int = 512
    dataset_train_path = "../data/ftdataset_train.tsv"
    dataset_val_path = "../data/ftdataset_val.tsv"
    dataset_test_path = "../data/ftdataset_test.tsv"
    logging_root_dir: str = "." # je sais pas pourquoi ça marche mais ça marche
    ckpt_root_dir: str = "./ckpt"

Documentation utilisée :
- https://huggingface.co/almanach/camembertv2-base
- https://arxiv.org/pdf/2411.08868
- https://www.datacamp.com/tutorial/fine-tuning-llama-3-1
- https://www.philschmid.de/fine-tune-modern-bert-in-2025
- https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/BERT/Fine_tuning_BERT_(and_friends)_for_multi_label_text_classification.ipynb
- lightning_topic_classification.py