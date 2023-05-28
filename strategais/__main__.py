import argparse
import os
import io
from .save_tools import *
import json
import asyncio
import uvicorn

from fastapi import FastAPI, Request, WebSocket  # ,Response, Body, Form,
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)

from pydantic import BaseModel
import pandas as pd
from pandas.io.json import json_normalize
import numpy as np
import requests

from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

import torch

import os
from dotenv import load_dotenv
from pathlib import Path
import pandas as pd
import joblib


parser = argparse.ArgumentParser(description='Start a Strategais FastAPI server.')
parser.add_argument('-t', '--title', default='Strategais Server', help='The title of the server.')
parser.add_argument('-d', '--description', default='Strategais Server', help='The description of the server.')
parser.add_argument('-p', '--port', type=int, default=8000, help='The port to serve the server on.')
parser.add_argument('-e', '--env', default='main.env', help='The .env file to load.')
args = parser.parse_args()

load_dotenv(args.env)

server = FastAPI(
    title=args.title,
    description=args.description)

server.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], )

@server.get("/keepalive")
def keepalive_get():
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {'timestamp': timestamp}
    
@server.get("/chat")
def chat_get(question: str = 'Hello World?'):
    model = joblib.load('small_model.sav')
    tokenizer = joblib.load('small_tokenizer.sav')

    pipe = pipeline(
        "text2text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=256,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.2,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    question = question
    with torch.no_grad(): 
        input_ids = tokenizer(question, return_tensors="pt").input_ids.to(torch.device('cpu'))
        outputs = model.generate(input_ids, max_length=10000)
        gpt_response = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return gpt_response

@server.websocket("/chat/ws")
async def chat_endpoint(websocket: WebSocket):
    model = joblib.load('small_model.sav')
    tokenizer = joblib.load('small_tokenizer.sav')

    pipe = pipeline(
        "text2text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=256,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.2,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    await websocket.accept()
    while True:
        question = await websocket.receive_text()
        template = """
        ### Instruction: 
        {instruction}

        Answer:"""

        prompt = PromptTemplate(template=template, input_variables=["instruction"])

        llm_chain = LLMChain(prompt=prompt, llm=local_llm)
        question = question
        gpt_response = llm_chain.run(question)
        await websocket.send_text(gpt_response)

if __name__ == "__main__":
    uvicorn.run(server, host="0.0.0.0", port=args.port, log_level="info")
