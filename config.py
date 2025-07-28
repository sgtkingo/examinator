import json
import os
import ast
from utils import to_bool

ALLOWED_EXTENSIONS = os.environ.get("ALLOWED_EXTENSIONS", ['.pdf', '.jpg', '.docx', '.doc', '.odt', '.txt', '.md'])
ALLOWED_EXTENSIONS = ast.literal_eval(str(ALLOWED_EXTENSIONS))
HARVEST_AS_IMAGE = to_bool(os.environ.get("HARVEST_AS_IMAGE", 'False'))
DOCUMENT_HARVEST_METHOD = os.environ.get('DOCUMENT_HARVEST_METHOD')
HARVEST_OPENAI_MODEL = os.environ.get('HARVEST_OPENAI_MODEL')

PDF_CONVERTOR_ENDPOINT = os.environ["PDF_CONVERTOR_ENDPOINT"]
DOCUMENT_IMAGE_RENDER_SCALE = float(os.environ.get("DOCUMENT_IMAGE_RENDER_SCALE", '2.0'))
DOCUMENT_IMAGE_RENDER_FORMAT = str(os.environ.get("DOCUMENT_IMAGE_RENDER_FORMAT", 'PNG'))
DOCUMENT_IMAGE_RENDER_DPI = int(os.environ.get("DOCUMENT_IMAGE_RENDER_DPI", '300'))

PLAIN_TEXT_MIN_LENGTH = int(os.getenv('PLAIN_TEXT_MIN_LENGTH', '200'))
PLAIN_TEXT_MAX_LENGTH = int(os.getenv('PLAIN_TEXT_MAX_LENGTH', '12000'))

GCP_SERVICE_ACCOUNT_CREDENTIALS = json.loads(os.environ['GCP_SERVICE_ACCOUNT_CREDENTIALS'])
GCP_DOCUMENTAI_PROCESSOR_NAME = os.environ['GCP_DOCUMENTAI_PROCESSOR_NAME']

LLM_MERGE = to_bool(os.getenv('LLM_MERGE', 'False'))

PLAINTEXT_MERGE_TEMPERATURE = float(os.getenv('PLAINTEXT_MERGE_TEMPERATURE', '0.0'))
PLAINTEXT_MERGE_OPENAI_LLM_MODEL = os.getenv('PLAINTEXT_MERGE_OPENAI_LLM_MODEL')

LLM_MERGE_METHOD = os.getenv('LLM_MERGE_METHOD')

AZURE_OPENAI_ENDPOINT = os.environ.get('AZURE_OPENAI_ENDPOINT', None)
AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY', None)
AZURE_OPENAI_LLM_DEPLOYMENT = os.environ.get('AZURE_OPENAI_LLM_DEPLOYMENT', None)

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', None)

TRUE_DOCUMENTS_DIR = 'input'

LLM_MIN_EXAMPLE_SIMILARITY = float(os.getenv('LLM_MIN_EXAMPLE_SIMILARITY', '0.0'))
LLM_MAX_EXAMPLE_SIMILARITY = float(os.getenv('LLM_MAX_EXAMPLE_SIMILARITY', 'inf'))

LLM_SYSTEM_PROMPT = """You are helpful assistant skilled in data extraction.
Your task is to extract information from the provided expense document."""

LLM_USER_PROMPT = """Extract attributes in the expense document mentioned below.
ExpenseDocument: {document}
"""

LLM_MAX_EXAMPLES = int(os.getenv('LLM_MAX_EXAMPLES', '2'))
PDF_MAX_PAGES = int(os.getenv('PDF_MAX_PAGES', '5'))

OPENAI_EMBEDDING_MODEL = os.environ['OPENAI_EMBEDDING_MODEL']

OPENAI_DOCUMENT_TYPE_LLM_MODEL = os.environ.get('OPENAI_DOCUMENT_TYPE_LLM_MODEL', 'gpt-4o-mini')
TESSERACT_PATH = os.getenv('TESSERACT_PATH', r'/usr/bin/tesseract')


# Validate configuration settings before using them
def validate_config() -> None:
    if LLM_MERGE:
        assert PLAINTEXT_MERGE_OPENAI_LLM_MODEL, "PLAINTEXT_MERGE_OPENAI_AI_LLM_MODEL is required"
        assert LLM_MERGE_METHOD, "LLM_MERGE_METHOD is required"

        if LLM_MERGE_METHOD == "azure_openai":
            assert AZURE_OPENAI_ENDPOINT, "AZURE_OPENAI_ENDPOINT is required"
            assert AZURE_OPENAI_API_KEY, "AZURE_OPENAI_API_KEY is required"
            assert AZURE_OPENAI_LLM_DEPLOYMENT, "AZURE_OPENAI_LLM_DEPLOYMENT is required"
        elif LLM_MERGE_METHOD == "openai":
            assert OPENAI_API_KEY, "OPENAI_API_KEY is required"


def export_config() -> dict[str, str | float]:
    config = {'HARVEST_AS_IMAGE': HARVEST_AS_IMAGE, 'LLM_MERGE': LLM_MERGE}

    if LLM_MERGE:
        config['PLAINTEXT_MERGE_OPENAI_LLM_MODEL'] = PLAINTEXT_MERGE_OPENAI_LLM_MODEL
        config['LLM_MERGE_METHOD'] = LLM_MERGE_METHOD
        config['PLAINTEXT_MERGE_TEMPERATURE'] = PLAINTEXT_MERGE_TEMPERATURE

        if LLM_MERGE_METHOD == "azure_openai":
            config['AZURE_OPENAI_ENDPOINT'] = AZURE_OPENAI_ENDPOINT
            config['AZURE_OPENAI_LLM_DEPLOYMENT'] = AZURE_OPENAI_LLM_DEPLOYMENT

    config['DOCUMENT_HARVEST_METHOD'] = DOCUMENT_HARVEST_METHOD
    config['HARVEST_OPENAI_MODEL'] = HARVEST_OPENAI_MODEL

    return config
