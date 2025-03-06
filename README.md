# YouTube Video Q&A

## Overview
YouTube Video Q&A is an AI-powered application that allows users to ask questions about a YouTube video and receive answers based on its transcript. The system extracts the video transcript, processes it into a searchable format, and retrieves relevant information to answer user queries using a Retrieval-Augmented Generation (RAG) approach.

## Features
- Extracts transcripts from YouTube videos.
- Splits transcripts into searchable chunks.
- Uses FAISS for efficient similarity search.
- Integrates with a language model (LLama-3.3-70B) for generating responses.
- Provides a Gradio-based interface for easy interaction.

## Installation
### Prerequisites
Ensure you have Python installed along with the required libraries.

```sh
pip install gradio langchain_community langchain_groq youtube_transcript_api faiss-cpu textwrap
```

## Usage
Run the application using the following command:

```sh
python app.py
```

Then, open the Gradio interface and enter:
1. A YouTube video URL.
2. A question related to the video's content.

The application will process the transcript and provide an answer based on the video's content.

## Code Explanation
- **Extract Transcript**: The `YoutubeLoader` retrieves the transcript of the provided YouTube video.
- **Text Splitting**: The `RecursiveCharacterTextSplitter` breaks the transcript into chunks to enhance search accuracy.
- **Vector Database**: FAISS is used to store and retrieve relevant transcript sections.
- **RAG-based Retrieval**: The model uses Retrieval-Augmented Generation (RAG) to find relevant transcript parts and generate responses.
- **LLM Response Generation**: A `ChatGroq` model processes the transcript and generates answers.
- **Gradio Interface**: Allows users to interact with the system easily.

## Example
```sh
YouTube Video URL: https://www.youtube.com/watch?v=example
Your Query: What is the main topic of the video?
Response: The video discusses...
```

## License
This project is open-source and available under the MIT License.

## Disclaimer
The accuracy of answers depends on the quality and completeness of the video transcript. If the transcript lacks sufficient details, the model may not provide an accurate response.

![Screenshot 2025-03-06 202539](https://github.com/user-attachments/assets/ff4a03b2-df09-4175-ac7c-380799ff5ef3)


