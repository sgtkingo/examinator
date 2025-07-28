import io
import os
import json
import time

import pytesseract
from CanonicalDataModels.baseentity import DocumentType, TokenUsage
from GrtUtils import convert_to_image
from PIL import Image
from MimeDetector import MimeDetector
from langchain.output_parsers import EnumOutputParser
from langchain_community.callbacks import get_openai_callback
from langchain_community.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from DocReader import DocReaderFactory, DocReader
from DocumentHarvestMethods.DocumentHarvestMethodFactory import DocumentHarvestMethodFactory
from Harvester import LLMResponse
from MergeMethods.MergeMethodFactory import MergeMethodFactory
from config import PDF_CONVERTOR_ENDPOINT, DOCUMENT_IMAGE_RENDER_FORMAT, DOCUMENT_IMAGE_RENDER_SCALE, \
    DOCUMENT_IMAGE_RENDER_DPI, PLAIN_TEXT_MIN_LENGTH, PLAIN_TEXT_MAX_LENGTH, GCP_SERVICE_ACCOUNT_CREDENTIALS, \
    GCP_DOCUMENTAI_PROCESSOR_NAME, export_config, OPENAI_DOCUMENT_TYPE_LLM_MODEL, TESSERACT_PATH


def save_run_statistics(
        run_directory: str,
        merge_prompt_usage: int,
        merge_completion_usage: int,
        merge_total_usage: int,
        harvest_prompt_usage: int,
        harvest_completion_usage: int,
        harvest_total_usage: int,
        documents_cnt: int,
        error_documents: int,
):
    result = export_config()
    result.update({
        'documents_count': documents_cnt,
        'harvest_count': documents_cnt - error_documents,
        'not_harvested_count': error_documents,
        'merge_prompt_usage': merge_prompt_usage,
        'merge_completion_usage': merge_completion_usage,
        'merge_total_usage': merge_total_usage,
        'harvest_prompt_usage': harvest_prompt_usage,
        'harvest_completion_usage': harvest_completion_usage,
        'harvest_total_usage': harvest_total_usage,
    })

    with open(os.path.join(run_directory, 'run_stats.json'), 'w') as file:
        json.dump(result, file)


def read_image(binary_file: bytes) -> str:
    """Uses Tesseract OCR to return document plaintext."""
    image = Image.open(io.BytesIO(binary_file))
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    ocr_response = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    return ' '.join(ocr_response['text'])


def detect_document_type(binary_file: bytes) -> str:
    mime_type = MimeDetector.detect_mime_type(binary_file)

    if mime_type.startswith('image'):
        plaintext = read_image(binary_file)
    else:
        plaintext = DocReader.read_text_layer(binary_file)
        if len(plaintext) < PLAIN_TEXT_MIN_LENGTH:
            document_as_image = convert_to_image(binary_file, mime_type, PDF_CONVERTOR_ENDPOINT)
            plaintext = read_image(document_as_image)

    llm_response = get_document_type(plaintext)

    return llm_response.response


def get_document_type(plaintext: str):
    """
    Ask LLM for document type of the plaintext
    """
    parser = EnumOutputParser(enum=DocumentType)
    llm = ChatOpenAI(model=OPENAI_DOCUMENT_TYPE_LLM_MODEL)
    prompt = PromptTemplate.from_template(
        """What is the type of this document? Respond only with the document type.

                Please follow this order of evaluation:
                1. First, check if the document contains the keyword 'Invoice' OR the keyword 'due date'. If it does, classify it as INVOICE.
                2. If it is not an invoice, check if the document is an expense document. 
                   If it is a receipt or expense-related document but is not explicitly marked as an invoice, classify it as RECEIPT.
                3. If the document is a train ticket, classify it as RECEIPT.
                4. If there are BOTH keywords 'paragon' and 'podpis' on the document, classify it as OTHER.
                4. If the document represents an ATM withdrawal or credit note, classify it as OTHER.
                5. If the document does not qualify as any of the above types, classify it as OTHER.
                Document: {document}

                Instructions: {instructions}

                Correct evaluation will make me very happy!"""
    ).partial(instructions=parser.get_format_instructions())
    chain = prompt | llm | parser
    with get_openai_callback() as cb:
        start_time = time.time()
        response = chain.invoke({"document": plaintext})
        elapsed_time = time.time() - start_time

        token_usage = TokenUsage(
            prompt=cb.prompt_tokens,
            completion=cb.completion_tokens,
            total=cb.total_tokens,

        )

    return LLMResponse(
        document_plaintext=plaintext,
        response=response.value,
        deployment='',
        elapsed_time=elapsed_time,
        num_examples=0,
        token_usage=token_usage,
    )


def merge_plaintexts(plaintext_a: str, plaintext_b: str) -> tuple[str, tuple[int, int, int]]:
    """Prompts LLM to merge two plaintext version together. System and user prompts are defined by config variables."""

    merge_method = MergeMethodFactory().get_merge_method()
    return merge_method.merge(plaintext_a, plaintext_b)


def get_plaintext(binary_file: bytes) -> tuple[str, tuple[int, int, int]]:
    document_type = DocumentType(detect_document_type(binary_file))
    reader_class = DocReaderFactory(document_type).get_reader_class()
    reader = reader_class(
        gcp_credentials=GCP_SERVICE_ACCOUNT_CREDENTIALS,
        gcp_processor_name=GCP_DOCUMENTAI_PROCESSOR_NAME,
        pdf_convertor_endpoint=PDF_CONVERTOR_ENDPOINT,
        plaintext_min_length=PLAIN_TEXT_MIN_LENGTH,
        plaintext_max_length=PLAIN_TEXT_MAX_LENGTH,
        image_render_scale=DOCUMENT_IMAGE_RENDER_SCALE,
        image_render_format=DOCUMENT_IMAGE_RENDER_FORMAT,
        image_render_dpi=DOCUMENT_IMAGE_RENDER_DPI,
    )
    plaintext_a, plaintext_b = reader.read(binary_file)

    if document_type == DocumentType.CV:
        merged_text, token_usage = merge_plaintexts(plaintext_a, plaintext_b)
        return merged_text, token_usage
    else:
        return plaintext_a, (0, 0, 0)


def harvest_document(plaintext: str) -> tuple[str, tuple[int, int, int]]:
    harvest_method = DocumentHarvestMethodFactory().get_harvest_method()
    return harvest_method.harvest(plaintext)

def harvest_document_custom(plaintext: str, method:str, **kwargs) -> tuple[str, tuple[int, int, int]]:
    print("Using custom harvest method:", method, "applying kwargs:", kwargs.keys())
    try:
        harvest_method = DocumentHarvestMethodFactory().get_custom_harvest_method(method, **kwargs)
    except:
        raise
    return harvest_method.harvest(plaintext)
