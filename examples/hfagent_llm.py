from strategais.save_tools import *
from strategais.llm_tools import *

from transformers import *
from transformers import GenerationMixin

def main_chat(question: str = 'Hello World?'):
        import requests
        import json

        url = "https://api-inference.huggingface.co/models/OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5"

        payload = json.dumps({
        "inputs": question
        })

        headers = {
        'Authorization': 'Bearer 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',
        'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)

        print(response.text)
        return response.text