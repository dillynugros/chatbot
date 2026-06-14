import streamlit as st
import os
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# KONFIGURASI TAMPILAN HALAMAN (UI/UX)
# ==========================================
st.set_page_config(
    page_title="Asisten Layanan DJPb",
    page_icon="🏛️",
    layout="centered"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #f4f6f9;
    }
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    .stChatMessage[data-baseweb="block"]:nth-child(even) {
        background-color: #e3f2fd; 
    }
    .stChatMessage[data-baseweb="block"]:nth-child(odd) {
        background-color: #ffffff; 
        border: 1px solid #bbdefb;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ Asisten Virtual Layanan Perbendaharaan")
st.caption("Layanan Informasi Berbasis Referensi Resmi - Kanwil DJPb")

# ==========================================
# LOGIKA RAG (Retrieval-Augmented Generation)
# ==========================================

@st.cache_resource(show_spinner="Menyiapkan otak AI dan menyinkronkan dokumen (mohon tunggu beberapa menit)...")
def load_and_process_documents():
    if not os.path.exists("referensi"):
        os.makedirs("referensi")
        return None

    loader = PyPDFDirectoryLoader("referensi/")
    documents = loader.load()
    
    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    # Model embedding terbaru yang valid
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    batch_size = 90 
    vectorstore = None
    
    progress_text = "Membaca dan memproses halaman regulasi..."
    my_bar = st.progress(0, text=progress_text)
    total_batches = (len(docs) + batch_size - 1) // batch_size
    
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        
        # MEKANISME ANTI-GAGAL (RETRY & BACKOFF) UNTUK API GRATIS
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_documents(batch, embeddings)
                else:
                    vectorstore.add_documents(batch)
                break # Jika berhasil, keluar dari loop retry
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(10) # Jika kena limit Google, diam 10 detik lalu coba lagi
                else:
                    # Jika gagal total 3 kali, tampilkan error aslinya ke layar admin
                    st.error(f"Gagal memproses dokumen pada batch ke-{i//batch_size + 1}. Error sistem: {str(e)}")
                    st.stop()
        
        current_batch = (i // batch_size) + 1
        progress_percentage = current_batch / total_batches
        my_bar.progress(progress_percentage, text=f"{progress_text} ({int(progress_percentage*100)}%) - Aman dari Limit API")
        
        # Jeda 5 detik antar batch = Maksimal 12 request per menit (Sangat aman untuk Free Tier Google API)
        time.sleep(5) 
            
    my_bar.empty() 
    return vectorstore

def get_conversational_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0) 
    
    system_prompt = (
        "Anda adalah Asisten Layanan yang profesional dan responsif. "
        "Gunakan HANYA potongan konteks referensi berikut untuk menjawab pertanyaan pengguna layanan (satker). "
        "Jika jawabannya tidak ada dalam konteks, Anda WAJIB menjawab: "
        "'Mohon maaf, berdasarkan dokumen referensi yang saya miliki, informasi tersebut tidak ditemukan. Silakan konsultasikan lebih lanjut dengan petugas CSO kami.' "
        "Dilarang keras menggunakan pengetahuan dari luar konteks, berasumsi, atau mengarang jawaban.\n\n"
        "Konteks referensi:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) 
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# ==========================================
# ANTARMUKA CHATBOT STREAMLIT
# ==========================================

if "GOOGLE_API_KEY" not in st.secrets:
    st.error("⚠️ API Key belum dikonfigurasi. Silakan atur GOOGLE_API_KEY di Streamlit Secrets.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Halo! Ada yang bisa saya bantu terkait aturan dan layanan perbendaharaan hari ini?"}]

vectorstore = load_and_process_documents()

if vectorstore is None:
    st.warning("Belum ada dokumen referensi. Silakan unggah file PDF aturan ke folder 'referensi/' di GitHub.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ketik pertanyaan Anda di sini..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Mencari referensi aturan yang relevan..."):
                try:
                    rag_chain = get_conversational_chain(vectorstore)
                    response = rag_chain.invoke({"input": prompt})
                    answer = response["answer"]
                    st.markdown(answer)
                    
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat mencari jawaban: {e}")
