from typing import Any
from langchain_community.document_loaders.html import UnstructuredHTMLLoader
from pydantic import BaseModel
from unstructured.partition.auto import partition
# raw_pdf_elements = partition(
#     filename="sec.htm",
# )
# for element in raw_pdf_elements:
#     if "unstructured.documents.elements.Table" in str(type(element)):
#         print(element)
# print(raw_pdf_elements)
# loader=UnstructuredHTMLLoader('sec.htm',mode="single")
# docs=loader.load()
# content=docs[0].page_content
# print(content+'1234')
# import tarfile
# file =tarfile.open('apple.tar')
# file.extractall('./extracts')
# file.close()
from langchain_google_community import TextToSpeechTool
text_to_speak = "Hello world!"

tts =TextToSpeechTool()
tts.name
speech_file = tts.run(text_to_speak)

