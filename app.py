import gradio as gr
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
import textwrap

embeddings = HuggingFaceEmbeddings()

def create_db_from_youtube_video_url(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    db = FAISS.from_documents(docs, embeddings)
    return db

def get_response_from_query(video_url, query, k=4):
    db = create_db_from_youtube_video_url(video_url)
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    chat = ChatGroq(api_key="your groq api", model="llama-3.3-70b-versatile", temperature=0)

    template = """You are a helpful assistant that can answer questions about YouTube videos
    based on the video's transcript: {docs}

    Only use the factual information from the transcript to answer the question.
    If you feel like you don't have enough information to answer the question, say "I don't know"."""

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "Answer the following question: {question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")

    return textwrap.fill(response, width=70)

def gradio_interface(video_url, query):
    return get_response_from_query(video_url, query)

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[gr.Textbox(label="YouTube Video URL"), gr.Textbox(label="Your Query")],
    outputs=gr.Textbox(label="Response"),
    title="YouTube Video Q&A",
    description="Enter a YouTube video URL and a question to get answers based on its transcript."
)

iface.launch()