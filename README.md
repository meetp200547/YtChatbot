
# YtChatbot: YouTube RAG Plugin

YtChatbot is a Streamlit-based web application that allows users to interact with YouTube videos through natural language queries. By providing a YouTube URL, the application extracts the video's transcript and leverages Google's Gemini AI models to accurately answer questions based exclusively on the video's content.

## Project Architecture and Data Flow

This project implements a Retrieval-Augmented Generation (RAG) architecture. The data flow follows these steps:

1. **User Input:** The user provides a YouTube URL via the Streamlit interface.
2. **Transcript Extraction:** The backend extracts the video ID and uses the `youtube-transcript-api` to download the corresponding transcript.
3. **Text Chunking:** To handle long transcripts effectively, the text is split into smaller, overlapping chunks (1000 characters each) using LangChain's `RecursiveCharacterTextSplitter`.
4. **Vector Embedding:** These text chunks are passed to Google's `gemini-embedding-2` model, which converts them into numerical vectors (embeddings). These vectors are then stored in a local FAISS vector database.
5. **Retrieval:** When a user submits a question, the query is converted into an embedding. The FAISS database performs a similarity search to retrieve the top 5 transcript chunks most relevant to the question.
6. **Answer Generation:** The retrieved transcript chunks are provided as context to the `gemini-3-flash-preview` model, along with the user's original question. The model analyzes the context and formulates a precise, context-aware response.

## Local Installation and Setup

1. Clone this repository and navigate to the project directory:
   ```bash
   git clone https://github.com/meetp200547/YtChatbot.git
   cd YtChatbot
   ```

2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory and add your Google Gemini API key:
   ```env
   GEMINI_API_KEY="your_api_key_here"
   ```

4. Launch the Streamlit application:
   ```bash
   streamlit run main.py
   ```

 Technology Stack

* Frontend:** Streamlit
* Orchestration:** LangChain
* Embeddings & LLM:** Google Gemini API (`gemini-embedding-2` and `gemini-3-flash-preview`)
* Vector Store:** FAISS (Facebook AI Similarity Search)

