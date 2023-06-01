import argparse
import os
import io
import importlib.util
from .save_tools import *
from .llm_tools import *
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

from transformers import *
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
parser.add_argument('-l', '--llm', help='The path to the Python file that defines the LLM to use.')

args = parser.parse_args()

load_dotenv(args.env)

from huggingface_hub import login
login(os.getenv('HUGGINGFACEHUB_API_TOKEN'))

if args.llm:
    spec = importlib.util.spec_from_file_location("llm", args.llm)
    llm_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(llm_module)
    chatbot = llm_module.main_chat()
else:
    def main_chat(question):
        from transformers.tools import HfAgent
        agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")
        return agent.run(question)
    
    chatbot = main_chat

server = FastAPI(
    title=args.title,
    description=args.description)

server.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], )

@server.get("/keepalive",
            tags=['Keep Alive API'],
            summary="Keep Alive API",
            description="Returns the server timestamp.")
def keepalive_get():
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {'timestamp': timestamp}
    
@server.get("/chat",
            tags=['Chat API'],
            summary="Chat API",
            description="Returns the chat response to the question.")
def chat_get(question: str = 'Hello World?'):
    return {"answer": chatbot(question)}

@server.websocket("/chat/ws")
async def chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        question = await websocket.receive_text()
        await websocket.send_text(chatbot(question))

import urllib3
urllib3.disable_warnings()


import pkg_resources

template_path = pkg_resources.resource_filename('strategais', 'templates')
templates = Jinja2Templates(directory=template_path)


@server.get("/", response_class=HTMLResponse,
            tags=['Landing API'],
            summary="Chat Home Page",
            description="Chat Home Page")
def index_html(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(server, host="0.0.0.0", port=args.port, log_level="info")
