from strategais.save_tools import *
from strategais.llm_tools import *

from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

import torch

def main_chat(question: str = 'Hello World?'):
        from transformers.tools import HfAgent
        agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
        return agent.chat(question)