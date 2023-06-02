from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='strategais',
    version='0.2.6', 
    description='A Python library for deploying large language models (LLMs) in local environments.',
    long_description=long_description,
    long_description_content_type="text/markdown", 
    packages=find_packages(),
    package_data={'strategais': ['templates/*.html', 'static/*.js', 'static/*.css' , 'models/*.sav']},
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
        'psycopg2-binary',
        'watchdog[watchmedo]',
        'email-validator',
        'boto3',
        'jinja2',
        'python-multipart',
        'httpx',
        'sse-starlette',
        'langchain',
        'python-dotenv',
        'openai',
        'huggingface_hub',
        'chromadb',
        'redis',
        'sentence_transformers',
        'unstructured',
        'pydantic',
        'datasets', 
        'loralib', 
        'sentencepiece', 
        'transformers',
        'pytorch-lightning',
        'torch',
        'accelerate',
        'torchaudio',
        'cryptography==38.0.4',
        'xformers==0.0.12'
    ],
)

