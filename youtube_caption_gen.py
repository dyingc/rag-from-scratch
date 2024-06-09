from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParser, FasterWhisperParser
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from typing import List, Any
from dotenv import load_dotenv
import os

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
    os.environ['http_proxy'] = ''
    os.environ['https_proxy'] = ''
    os.environ['no_proxy'] = ''

def get_beautify_prompt():
    system = """You are an excellent editor. Your task is to review and edit a text transcribed from an audio clip. Your duties include:

	•	Understanding the text and punctuating it
	•	Segmenting the text into paragraphs (separate paragraphs using "\\n")
	•	Correcting the obvious errors or typos that may arise due to the transcription from audio
    
    You should keep the original language, without "translation". For example, if the input language is French, the output should also be French, if the input is Simplified Chinese, so as the output language. No translation is allowed!

    {format_instruction}
    """

    human = """Audio Transcript:\n{audio_doc}"""
    text_schema = ResponseSchema(name="refined_text", type="string", description="The refined text - Note: you should not change or translate the original language. So English in, English out; French in, French out")
    language_schema = ResponseSchema(name="language", type="string", description="The detected language of the text")
    schemas = [text_schema, language_schema]
    output_parser = StructuredOutputParser(name="refined_text_output_parser", response_schemas=schemas)
    template = f"""{system}\n\n{human}"""
    prompt = PromptTemplate.from_template(template)
    prompt.partial_variables = {"format_instruction": output_parser.get_format_instructions()}
    prompt.input_variables = ["audio_doc"]
    return prompt, output_parser

def get_beautify_llm(free:bool=False):
    from langchain_groq import ChatGroq
    from langchain_openai import ChatOpenAI
    groq = ChatGroq(name = "groq_to_beautify", model="llama3-8b-8192", max_tokens=8192, temperature=0)
    openai = ChatOpenAI(name="openai_to_beautify", verbose=True, model="gpt-3.5-turbo", temperature=0, max_tokens=4096)
    if free:
        return groq
    else:
        return openai

def beautify_audio_doc(audio_doc:str, chain_name:str="Youtube Caption Beautify"):
    prompt, output_parser = get_beautify_prompt()
    FREE_LLM = os.environ.get('FREE_LLM', None)
    if FREE_LLM is not None and FREE_LLM.lower() == 'true':
        free = True
    else:
        free = False
    llm = get_beautify_llm(free=free)
    def additional_processing(x: Any):
        x.content = x.content.replace('"""\n', '"')
        x.content = x.content.replace('"""', '"')
        return x
    from langchain_core.runnables import RunnableLambda
    chain = prompt | llm | RunnableLambda(func=additional_processing) | output_parser
    chain.name = chain_name
    response = chain.invoke({"audio_doc": audio_doc})
    return response

def beautify_audio_docs(audio_docs:List[str]):
    docs = list()
    for doc in audio_docs:
        beautify_chain_name = f"Youtube Caption Beautify: {audio_docs[0].dict().get('metadata').get('source')} - {audio_docs[0].dict().get('metadata').get('chunk')}"
        response = beautify_audio_doc(doc.page_content, chain_name=beautify_chain_name)
        refined_doc = Document(page_content=response.get('refined_text'), metadata={"source": doc.metadata.get("source"), "language": response.get("language")})
        docs.append(refined_doc)
    return docs

def main(youtube_urls:List[str], save_folder:str):
    load_dotenv()
    os.environ['LANGCHAIN_PROJECT'] = 'Youtube Caption Generation'
    os.environ['FREE_LLM'] = 'True'
    set_proxy()
    audio_loader = gen_audio_loader(youtube_urls, save_folder)
    blob_parser = gen_blob_parser()
    audio_docs = parse(audio_loader, blob_parser=blob_parser)
    # audio_docs = list()
    # with open("/private/tmp/yt_audios/test.txt", "r") as f:
    #     audio_docs.append(''.join(f.readlines()))
    # audio_docs = [Document(page_content=doc, metadata={"source": "RAG from scratch: Part 11"}) for doc in audio_docs]
    beautified_audio_docs = beautify_audio_docs(audio_docs)
    write_audio_docs(beautified_audio_docs)
    print(f"Saved {len(audio_docs)} audio docs to {save_folder}")

if __name__ == "__main__":
    youtube_urls = ["https://www.youtube.com/watch?v=kl6NwWYxvbM&list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x&index=11"]
    save_folder = "/tmp/yt_audios"
    main(youtube_urls, save_folder)