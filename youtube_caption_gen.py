from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import OpenAIWhisperParser
from typing import List
from dotenv import load_dotenv

def gen_audio_loader(youtube_urls:List[str], save_folder:str):
    audio_loader = YoutubeAudioLoader(urls=youtube_urls, save_dir=save_folder)
    return audio_loader

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

def main(youtube_urls:List[str], save_folder:str):
    set_proxy()
    audio_loader = gen_audio_loader(youtube_urls, save_folder)
    audio_docs = parse(audio_loader, blob_parser=OpenAIWhisperParser())
    write_audio_docs(audio_docs, save_folder)
    print(f"Saved {len(audio_docs)} audio docs to {save_folder}")

if __name__ == "__main__":
    load_dotenv()
    youtube_urls = ["https://www.youtube.com/watch?v=qNtjoEb54Mg"]
    save_folder = "/tmp/yt_audios"
    main(youtube_urls, save_folder)