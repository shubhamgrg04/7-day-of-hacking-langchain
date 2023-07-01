# Author: Shubham Garg (shubhamgrg04@gmail.com)
# Description: Extracts stock recommendations from a YouTube video transcript.

import os
import logging
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Sequence, Union
from datetime import datetime

from langchain.callbacks import get_openai_callback
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import TokenTextSplitter
from pytube import YouTube
from youtube_transcript_api import NoTranscriptFound,TranscriptsDisabled

TRANSCRIPT_DIRECTORY = 'transcripts'
OUTPUT_DIRECTORY = 'output'
LLM_MODEL = "gpt-3.5-turbo-16k"
MODEL_CONFIGS = {
    "gpt-3.5-turbo": {
        "max_tokens": 4000,
        "pricing_per_1k": 0.0015
    },
    "gpt-3.5-turbo-16k": {
        "max_tokens": 16000,
        "pricing_per_1k": 0.003
    }
}

logging.basicConfig(level=logging.DEBUG)


def get_transcript_path(video_id):
    return f"{TRANSCRIPT_DIRECTORY}/{video_id}.txt"


def get_output_path(video_id):
    now = datetime.now().strftime("%d/%m/%Y:%H:%M:%S")
    return f"{OUTPUT_DIRECTORY}/{video_id}/{now}.txt"


def get_transcript_if_exist(url):
    transcript_path = get_transcript_path(YouTube(url).video_id)
    if os.path.exists(transcript_path):
        with open(transcript_path, 'r') as file:
            transcript_text = file.read()
            return transcript_text
    return None


def save_transcript(url, transcript_text):
    if not os.path.exists(TRANSCRIPT_DIRECTORY):
        os.makedirs(TRANSCRIPT_DIRECTORY)
    transcript_path = get_transcript_path(YouTube(url).video_id)
    with open(transcript_path, 'w+') as file:
        file.write(transcript_text)


def save_output(url, output_text):
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    output_path = get_output_path(YouTube(url).video_id)
    with open(output_path, 'w') as file:
        file.write(output_text)


def get_video_transcript(url:str) -> str:
    """
    Retrieves the transcript of a YouTube video.

    Args:
        url (str): The URL of the YouTube video.

    Returns:
        str: The transcript of the YouTube video.
    """

    if not url:
        return []

    # check if we already have the transcript for the video
    transcript_text = get_transcript_if_exist(url)
    if transcript_text:
        return transcript_text
    
    # check if youtube has a transcript for the video
    try:
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=True,
            language=["en", "en-US", "en-IN"]
        )
        transcripts = loader.load()
        if transcripts:
            transcript_text = transcripts[0].page_content
            save_transcript(url, transcript_text)
            return transcript_text
    except (NoTranscriptFound,TranscriptsDisabled) as err:
        logging.error(f"failed to fetch transcript from youtube, exception, falling back to generating transcript using whisper")

    # fallback to transcription using whisper
    with TemporaryDirectory() as tmpdir:
        yt = YouTube(url)
        video = yt.streams.filter(only_audio=True).first()
        out_file = video.download(output_path=tmpdir.name)
        
        import whisper
        logging.info(f"starting transcription of the audio file - {out_file}")
        model = whisper.load_model("medium")
        result = model.transcribe(audio=out_file, verbose=True, initial_prompt="")
        transcript_text = result['text'].strip()
        save_transcript(url, transcript_text)
        return transcript_text


def run(youtube_url):

    with get_openai_callback() as cb:

        openai_api_key = os.environ.get("OPENAI_API_KEY")

        # get transcript from video, and save to transcripts folder
        transcript_text = get_video_transcript(youtube_url)
        if not transcript_text:
            return

        # pre-process transcript before feeding to model
        model_config = MODEL_CONFIGS[LLM_MODEL]
        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name=LLM_MODEL, temperature=0.5)
        logging.info(f"token count of complete transcript - {llm.get_num_tokens(transcript_text)}")
        text_splitter = TokenTextSplitter(model_name=LLM_MODEL, chunk_size = model_config["max_tokens"]*4, chunk_overlap = 0)
        documents = text_splitter.split_documents([Document(page_content=transcript_text)])
        logging.info(f"transcript split into {len(documents)} documents")
        for i,doc in enumerate(documents):
            logging.info(f"token count for document {i} - {llm.get_num_tokens(doc.page_content)}")

        # get stock selection from model
        selection_template = """
            You are an expert in analyzing stocks based on there financial performance and market conditions.
            You have read extensively on long term value investing and have been very successful in picking good quality stocks.
            Your goal is to create a find potential stocks to invest in.
            The stock selection should be based on the following text:
            ------------
            {text}
            ------------
            Given the text, give your understanding and opinions on the analyzed stocks. The stock selection is aimed at maximizing financial returns by investing in undervalued stocks.
            The stock selection should be as detailed as possible.
            STOCK SELECTION:
        """
        PROMPT_SELECTION = PromptTemplate(template=selection_template, input_variables=["text"])

        selection_refine_template = """
            You are an expert in analyzing stocks based on there financial performance and market conditions.
            Your goal is to create a find potential stocks to invest in.
            We have provided an existing stock selection up to a certain point: {existing_answer}
            We have the opportunity to refine the stock selection
            (only if needed) with some more context below.
            ------------
            {text}
            ------------
            Given the new context, refine the stock selection.
            The stock selection is aimed at maximizing financial returns by investing in undervalued stocks.
            If the context isn't useful, return the original stock selection.
        """
        PROMPT_REFINE = PromptTemplate(template=selection_refine_template, input_variables=["existing_answer", "text"])

        selection_chain = load_summarize_chain(llm=llm, chain_type='refine', verbose=True, question_prompt=PROMPT_SELECTION, refine_prompt=PROMPT_REFINE)
        selection = selection_chain.run(documents)
        logging.debug(selection)


        # save results to outputs folder
        save_output(youtube_url, selection)
    
    # Print the total cost of the API calls.
    logging.info(cb)

run("https://www.youtube.com/watch?v=yMqoL0CWIUo")