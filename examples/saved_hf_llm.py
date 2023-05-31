from strategais.save_tools import *
from strategais.llm_tools import *

from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

import torch


def main_chat(question: str = 'Hello World?'):
    download_from_s3('bucket_name', 's3_prefix', 'local_dir')
    model = load_model('model.sav')
    tokenizer = load_tokenizer('tokenizer.sav')

    llm = get_llm(model, tokenizer)
    template = """
            ### Instruction: 
            {instruction}

            Answer:"""

    prompt = PromptTemplate(template=template, input_variables=["instruction"])

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    gpt_response = llm_chain.run(question)
    return gpt_response