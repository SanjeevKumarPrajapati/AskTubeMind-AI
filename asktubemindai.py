import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import re
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled,VideoUnavailable,NoTranscriptFound,CouldNotRetrieveTranscript
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser

HF_TOKEN = st.secrets["HF_TOKEN"]
embedding=HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    huggingfacehub_api_token=HF_TOKEN,
    task="text-generation"
)
model=ChatHuggingFace(llm=llm,temperature=0.7)

@st.cache_resource(show_spinner=False)
def build_youtube_chain(video_id):
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(
            video_id=video_id,
            languages=["en", "hi"]
        )
        transcript = " ".join(chunk.text for chunk in transcript_list)

    except (
        TranscriptsDisabled,
        VideoUnavailable,
        NoTranscriptFound,
        CouldNotRetrieveTranscript,
    ):
        return None
    
    except Exception:
        # Catch anything unexpected
        return None

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.create_documents([transcript])

    vector_store = FAISS.from_documents(chunks, embedding)

    retrieval = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer only from the provided transcript context.
        If the context is insufficient, say "I couldn’t find this information in the video transcript.  
        You may try asking something else related to the video.".

        Context:
        {context}

        Question:
        {question}
        """,
        input_variables=["context", "question"]
    )

    def format_doc(retrieved_docs):
        return "\n\n".join(doc.page_content for doc in retrieved_docs)

    chain = (
        RunnableParallel({
            "context": retrieval | RunnableLambda(format_doc),
            "question": RunnablePassthrough()
        })
        | prompt
        | model
        | StrOutputParser()
    )

    return chain

def youtube(video_id, question, chain):
    return chain.invoke(question)


st.set_page_config(page_title="YouTube RAG Chatbot", page_icon="🎥")

st.title("🎥 AskTubeMind")

# ---------------- Session State ----------------

if "chains" not in st.session_state:
    st.session_state.chains = {}  # {video_id: chain}

if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []
    

# ---------------- Utils ----------------

def extract_youtube_id(text):
    pattern = r"(?:v=|\/|youtu\.be\/|shorts\/)([0-9A-Za-z_-]{11})|([0-9A-Za-z_-]{11})"
    match = re.search(pattern, text)
    return match.group(1) or match.group(2) if match else None

# ---------------- Chat UI ----------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Paste YouTube link / ID or ask a question...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    video_id = extract_youtube_id(user_input)

    with st.chat_message("assistant"):
        # 🎯 New video detected
        if video_id and video_id != st.session_state.current_video_id:
            st.markdown("⏳ **New video detected. Loading transcript...**")

            chain = build_youtube_chain(video_id)

            if chain is None:
                response = (
                    "❌ **Unable to fetch transcript for this video.**\n\n"
                    "Possible reasons:\n"
                    "- Video is private or deleted\n"
                    "- Transcripts are disabled\n"
                    "- Invalid YouTube link or ID\n\n"
                    "Please try another video."
                    )
            else:
                st.session_state.chains[video_id] = chain
                st.session_state.current_video_id = video_id
                response = "✅ **Transcript is ready!** Ask your questions."

        # 💬 Question for existing video
        else:
            if st.session_state.current_video_id:
                chain = st.session_state.chains[
                    st.session_state.current_video_id
                ]
                with st.spinner("🤖 Thinking..."):
                    response = chain.invoke(user_input)
            else:
                response = "📌 Please provide a YouTube video link or ID first."

        st.markdown(response)
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

# ---------------- Sidebar ----------------

    
with st.sidebar:
    st.header("📊 Session Info")
    st.write("**Current Video ID**")
    st.code(st.session_state.current_video_id or "None")

    st.write("**Loaded Videos**")
    st.write(list(st.session_state.chains.keys()) or "None")
    


