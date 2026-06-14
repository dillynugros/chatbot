import streamlit as st
import os
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

# Kustomisasi warna (Monokromatik Biru khas Perbendaharaan)
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ Asisten Virtual Layanan Perbendaharaan")
st.caption("Kanwil DJPb - Layanan Informasi Berbasis Referensi Resmi")

# ==========================================
# LOGIKA RAG (Retrieval-Augmented Generation)
# ==========================================

# 1. Fungsi untuk memproses dokumen (Di-cache agar cepat)
@st.cache_resource(show_spinner="Membaca dokumen referensi...")
def load_and_process_documents():
    # Pastikan folder referensi ada
    if not os.path.exists("referensi"):
        os.makedirs("referensi")
        return None

    loader = PyPDFDirectoryLoader("referensi/")
    documents = loader.load()
    
    if not documents:
        return None

    # Memecah dokumen menjadi potongan kecil (chunks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    
    # Membuat Vector Database menggunakan FAISS dan Gemini Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    return vectorstore

# 2. Persiapan Chain (Prompt Ketat)
def get_conversational_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0) # Temperature 0 agar tidak kreatif/mengarang
    
    # Prompt super ketat
    system_prompt = (
        "Anda adalah Asisten Layanan Kanwil DJPb yang profesional dan ramah. "
        "Gunakan HANYA potongan konteks referensi berikut untuk menjawab pertanyaan pengguna. "
        "Jika jawabannya tidak ada dalam konteks, Anda WAJIB menjawab: "
        "'Mohon maaf, berdasarkan dokumen referensi yang saya miliki, informasi tersebut tidak ditemukan. Silakan konsultasikan lebih lanjut dengan petugas CSO kami.' "
        "Dilarang keras menggunakan pengetahuan dari luar konteks atau mengarang jawaban.\n\n"
        "Konteks:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4}) # Mengambil 4 potongan teks paling relevan
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# ==========================================
# ANTARMUKA CHATBOT STREAMLIT
# ==========================================

# Inisialisasi API Key dari Streamlit Secrets
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("API Key belum dikonfigurasi. Silakan atur GOOGLE_API_KEY di Streamlit Secrets.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Inisialisasi memori percakapan
if "messages" not in st.session_state:
    st.session_state.messages = []

# Proses dokumen di latar belakang
vectorstore = load_and_process_documents()

if vectorstore is None:
    st.warning("Belum ada dokumen referensi di folder 'referensi/'. Silakan unggah file PDF aturan (misal: PMK) ke repositori GitHub Anda.")
else:
    # Tampilkan riwayat chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Tangkap input dari pengguna layanan
    if prompt := st.chat_input("Tanyakan seputar aturan perbendaharaan..."):
        # Tambahkan ke memori dan tampilkan di UI
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Proses jawaban
        with st.chat_message("assistant"):
            with st.spinner("Mencari di dalam dokumen referensi..."):
                rag_chain = get_conversational_chain(vectorstore)
                response = rag_chain.invoke({"input": prompt})
                answer = response["answer"]
                st.markdown(answer)
                
        # Simpan jawaban ke memori
        st.session_state.messages.append({"role": "assistant", "content": answer})
