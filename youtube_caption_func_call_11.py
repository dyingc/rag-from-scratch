from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParser, FasterWhisperParser
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from typing import List, Any, Optional
from dotenv import load_dotenv
import os

from langchain_core.pydantic_v1 import BaseModel, Field

class RefinedText(BaseModel):
    """You are an excellent writor/editor. Your task is to review and provide information about the input text, including:
    
    1. Refine the text:
	•	Understanding the text and punctuating it
	•	Segmenting the text into paragraphs (separate paragraphs using "\\n")
	•	Correcting the obvious errors or typos that may arise due to the transcription from audio
    You should keep the original language, without "translation". For example, if the input language is French, the output should also be French, if the input is Simplified Chinese, so as the output language. No translation is allowed!
    
    2. Some other information including summary, keywords, and translation    
    """

    refined_text: str = Field(
        ..., 
        description="The refined text that has been rectified, punctuated and segmented into paragraphs (separate paragraphs using '\\n')"
    )

    language: str = Field(
        ..., 
        description="The detected language of the text"
    )

    keywords: List[str] = Field(
        None,
        description="Keywords extracted from the text"
    )

    summary: str = Field(
        None,
        description="Summary of the text"
    )

    translated_text: Optional[str] = Field(
        None,
        description="Translate the 'refined_text' into Simplified Chinese. None if the language of text is already Simplified Chinese"
    )
    
    translated_summary: Optional[str] = Field(
        None,
        description="Translate the 'summary' into Simplified Chinese. None if the language of text is already Simplified Chinese"
    )

    translated_keywords: Optional[List[str]] = Field(
        None,
        description="Translate the 'keywords' into Simplified Chinese. None if the language of text is already Simplified Chinese"
    )

    def pretty_print(self) -> None:
        for field in self.__fields__:
            if getattr(self, field) is not None and getattr(self, field) != getattr(
                self.__fields__[field], "default", None
            ):
                print(f"{field}: {getattr(self, field)}")



def gen_audio_loader(youtube_urls:List[str], save_folder:str):
    audio_loader = YoutubeAudioLoader(urls=youtube_urls, save_dir=save_folder)
    return audio_loader

def gen_blob_parser():
    import torch
    blob_parser = FasterWhisperParser() if torch.cuda.is_available() else OpenAIWhisperParser()
    return blob_parser

def parse(audio_loader:YoutubeAudioLoader, blob_parser=OpenAIWhisperParser()):
    audio_docs = GenericLoader(blob_loader=audio_loader, blob_parser=blob_parser).load()
    return audio_docs

def write_audio_docs(audio_docs:List[str]):
    for i, doc in enumerate(audio_docs):
        txt_file = f"{doc.dict().get('metadata').get('source')}_{i:02d}.txt"
        with open(txt_file, "a") as f:
            f.write(doc.page_content)

def set_proxy():
    import os
    
    # a function to test the connectivity to the proxy
    def get_proxy():
        ora_proxy = 'http://www-proxy-ash7.us.oracle.com:80'
        import requests
        try:
            requests.get(ora_proxy, timeout=1)
            return ora_proxy
        except Exception as e:
            return ""
    proxy = get_proxy()
    os.environ['http_proxy'] = proxy
    os.environ['https_proxy'] = proxy
    os.environ['no_proxy'] = proxy

def get_beautify_llm(free:str='False'):
    if free is not None and free.lower() == "true":
        from langchain_groq import ChatGroq
        groq = ChatGroq(name = "groq_to_beautify", model="llama3-8b-8192", max_tokens=8192, temperature=0)
        groq_structured_output = groq.with_structured_output(RefinedText)
        groq_structured_output.name = "groq_to_beautify_with_structured_output"
        return groq, groq_structured_output
    else:
        from langchain_openai import ChatOpenAI
        openai = ChatOpenAI(name="openai_to_beautify", verbose=True, model="gpt-3.5-turbo", temperature=0, max_tokens=4096)
        openai_structured_output = openai.with_structured_output(RefinedText)
        openai_structured_output.name = "openai_to_beautify_with_structured_output"
        return openai, openai_structured_output
    
def get_beautify_prompt():
    template = """You are an excellent editor. Your task is to review and edit a text transcribed from an audio clip. Your duties include:

	•	Understanding the text and punctuating it
	•	Segmenting the text into paragraphs (separate paragraphs using "\\n")
	•	Correcting the obvious errors or typos that may arise due to the transcription from audio
    
    Audio Transcript:
    {audio_doc}
    """
    prompt = PromptTemplate.from_template(template)
    prompt.input_variables = ["audio_doc"]
    prompt.name = "beautify_prompt"
    return prompt

def get_beautify_chain(chain_name:str="beautify_chain"):
    prompt = get_beautify_prompt()
    _, llm_structured_output = get_beautify_llm(free=os.environ.get("FREE_LLM", "True"))
    chain = prompt | llm_structured_output
    chain.name = chain_name
    return chain

def output_result(result:RefinedText, filename:str):
    for field in result.__fields__:
        if getattr(result, field) is not None and getattr(result, field) != getattr(
            result.__fields__[field], "default", None
        ):
            with open(filename, "a") as f:
                f.write(f"{'#' * 15}\n{field}\n{'#' * 15}\n\n{getattr(result, field)}\n\n")

def beautify_doc(doc:Document, chain_name:str=None):
    if chain_name is None:
        chain_name = f"Beautify: {doc.metadata.get('source')}"
    chain = get_beautify_chain(chain_name)
    response = chain.invoke({"audio_doc": doc.page_content})
    output_file = f"{doc.metadata.get('source')}.txt"
    output_result(response, output_file)
    # response.pretty_print()

def beautify_audio_docs(audio_docs:List[Document]):
    for doc in audio_docs:
        beautify_doc(doc)


def main(youtube_urls:List[str], save_folder:str):
    load_dotenv()
    os.environ['LANGCHAIN_PROJECT'] = 'Youtube Caption Generation'
    os.environ['FREE_LLM'] = 'False'
    set_proxy()

    audio_loader = gen_audio_loader(youtube_urls, save_folder)
    blob_parser = gen_blob_parser()
    audio_docs = parse(audio_loader, blob_parser=blob_parser)

    # audio_docs = list()
    # with open("/private/tmp/yt_audios/test.txt", "r") as f:
    #     audio_docs.append(''.join(f.readlines()))
    # audio_docs = [Document(page_content=doc, metadata={"source": "RAG from scratch: Part 11"}) for doc in audio_docs]

    beautify_audio_docs(audio_docs)
    print(f"Saved {len(audio_docs)} audio docs to {save_folder}")

if __name__ == "__main__":
    youtube_urls = ["https://www.youtube.com/watch?v=kl6NwWYxvbM&list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x&index=11"]
    save_folder = "/tmp/yt_audios"
    main(youtube_urls, save_folder)