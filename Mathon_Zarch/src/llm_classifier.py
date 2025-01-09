from jinja2 import Template
from ollama import Client
import re
import json

from transformers import pipeline

from config import Config


_PROMPT_TEMPLATE = """Considérez l'avis suivant:

"{{text}}"

Quelle est la valeur de l'opinion exprimée sur chacun des aspects suivants : Prix, Cuisine, Service, Ambiance?

La valeur d'une opinion doit être une des valeurs suivantes: "Positive", "Négative", "Neutre", ou "Non exprimée".

La réponse doit se limiter au format json suivant:
{ "Prix": opinion, "Cuisine": opinion, "Service": opinion, "Ambiance": opinion}."""



class LLMClassifier:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Instantiate an ollama client
        self.llmclient = Client(host=cfg.ollama_url)
        self.model_name = 'gemma2:2b'
        self.model_options = {
            'num_predict': 500,  # max number of tokens to predict
            'temperature': 0.1,
            'top_p': 0.9,
        }
        self.jtemplate = Template(_PROMPT_TEMPLATE)


    def predict(self, text: str) -> dict[str,str]:
        """
        Lance au LLM une requête contenant le texte de l'avis et les instructions pour extraire
        les opinions sur les aspects sous forme d'objet json
        :param text: le texte de l'avis
        :return: un dictionnaire python avec une entrée pour chacun des 4 aspects ayant pour valeur une des
        4 valeurs possibles pour l'opinion (Positive, Négative, Neutre et NE)
        """
        prompt = self.jtemplate.render(text=text)
        result = self.llmclient.generate(model=self.model_name, prompt=prompt, options=self.model_options)
        response = result['response']
        jresp = self.parse_json_response(response)
        return jresp

    def parse_json_response(self, response: str) -> dict[str, str] | None:
        m = re.findall(r"\{[^\{\}]+\}", response, re.DOTALL)
        if m:
            try:
                jresp = json.loads(m[0])
                for aspect, opinion in jresp.items():
                    if "non exprim" in opinion.lower():
                        jresp[aspect] = "NE"
                return jresp
            except:
                return None
        else:
            return None
























