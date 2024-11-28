import os
from openai import OpenAI

class Crossover:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ['CBM_POP_APIKEY'],
            organization='org-aHxs2hPTZTYEPh8GyXZaDPmI',
            project='proj_bMwxaxkpiKADKECQtHOCCUjh',
        )
        pass