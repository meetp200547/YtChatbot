import streamlit as st
import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.documents import Document
from dotenv import load_dotenv
import os


load_dotenv()

def extract_video_id(url: str) -> str:
    """Extract the YouTube video ID from a given URL."""
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    if len(url) == 11:
        return url
    return None

def get_transcript(video_id: str) -> str:
    """Fetch the transcript for a YouTube video."""
    try:
        transcript_obj = YouTubeTranscriptApi().fetch(video_id)
        return " ".join([t.text for t in transcript_obj.snippets])
    except TranscriptsDisabled:
        return "Transcript is disabled for this video."
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

st.set_page_config(page_title="YouTube Q&A")
st.title("YouTube Video Q&A with Gemini")

if not os.environ.get("GEMINI_API_KEY"):
    st.warning("Please add your GEMINI_API_KEY to the .env file to continue.")
    st.stop()


url = st.text_input("Enter YouTube Video URL:")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None

if url:
    video_id = extract_video_id(url)
    if not video_id:
        st.error("Invalid YouTube URL.")
    else:
        st.success(f"Video ID Extracted: {video_id}")
        
   
        if video_id != st.session_state.current_video_id:
            with st.spinner("Fetching transcript and processing..."):
                transcript = get_transcript(video_id)
                
                if "Error" in transcript or "disabled" in transcript:
                    st.error(transcript)
                    st.session_state.vectorstore = None
                else:
              
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_text(transcript)
                    documents = [Document(page_content=chunk, metadata={"video_id": video_id}) for chunk in chunks]

            
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
                    st.session_state.vectorstore = FAISS.from_documents(documents, embeddings)
                    st.session_state.current_video_id = video_id
                    st.success("Video processed successfully! You can now ask questions.")
        
     
        if st.session_state.vectorstore:
            st.divider()
            st.subheader("Ask a Question")
            query = st.text_input("What would you like to know about this video?")
            
            if query:
                with st.spinner("Generating answer..."):
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
                    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
                    
                    prompt = PromptTemplate.from_template(
                        "Answer the question based only on the following context from a YouTube video transcript:\n\n{context}\n\nQuestion: {input}\n\nAnswer:"
                    )
                    
                    qa_chain = create_stuff_documents_chain(llm, prompt)
                    rag_chain = create_retrieval_chain(retriever, qa_chain)
                    
                    response = rag_chain.invoke({"input": query})
                    
                    st.write("**Answer:**")
                    st.info(response["answer"])
