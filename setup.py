from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='strategais',
    version='0.1.2', 
    description='A Python library for deploying large language models (LLMs) in local environments.',
    long_description=long_description,
    long_description_content_type="text/markdown", 
    packages=find_packages(),
    install_requires=[
        'fastapi',
        'fastapi_utils',
        'uvicorn',
        'starlette',
        'websockets',
        'flask',
        'numpy',
        'pandas',
        'requests',
        'scikit-learn',
        'scipy',
        'SQLAlchemy',
        'pydantic',
        'aiofiles',
        'openpyxl',
        'xlrd==1.2.0',
        'psycopg2-binary',
        'watchdog[watchmedo]',
        'email-validator',
        'boto3',
        'jinja2',
        'python-multipart',
        'httpx',
        'neo4j',
        'sse-starlette',
        'langchain',
        'python-dotenv',
        'openai',
        'huggingface_hub',
        'chromadb',
        'redis',
        'streamlit',
        'streamlit-pills==0.3.0',
        'sentence_transformers',
        'unstructured',
        'pydantic',
        'xformers',
        'datasets', 
        'loralib', 
        'sentencepiece', 
        'transformers',
        'pytorch-lightning',
        'torch'
    ],
)

