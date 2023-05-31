from transformers import pipeline
from langchain.llms import HuggingFacePipeline

def get_llm(model, tokenizer):
    pipe = pipeline(
        "text2text-generation",
        model=model, 
        tokenizer=tokenizer, 
        max_length=256,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.2,
    )
    return HuggingFacePipeline(pipeline=pipe)