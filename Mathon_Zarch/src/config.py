from dataclasses import dataclass

# NE PAS MODIFIER CE FICHIER, si voulez utiliser d'autres valeurs pour ces parmètre de config, faites-le
# sur la ligne de commande
# par exemple: pour lancer tester votre rendu programme  avec 2 runs et en utilisant la gpu
# numéro 0, il suffit de taper la commande suivante:
#
#           python runproject.py --n_runs=2 --device=0
#

@dataclass
class Config:
    # General options
    device: int = -1
    ollama_url: str = 'http://localhost:11434'
    n_runs: int = 5
    # n_train is the number of samples on which to run the eval. n_trian=-1 means eval on all test data,
    n_train: int = -1
    # n_test is the number of samples on which to run the eval. n_test=-1 means eval on all test data,
    n_test: int = -1



