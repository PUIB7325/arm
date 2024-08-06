import streamlit as st
from openai import AzureOpenAI
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import config  # Importer le fichier de configuration
import fitz  # PyMuPDF
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

# Configuration du client Azure OpenAI
client = AzureOpenAI(
    api_key=config.API_KEY,
    api_version=config.API_VERSION,
    azure_endpoint=config.AZURE_ENDPOINT
)

# Initialisation du modèle de embedding et de la base de données FAISS
@st.cache_resource
def init_embedding_model_and_index():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = faiss.IndexFlatL2(384)  # 384 est la dimension des embeddings pour ce modèle
    return model, index

model, index = init_embedding_model_and_index()

# Fonction pour ajouter des documents à la base de données
def add_documents(documents):
    embeddings = model.encode(documents)
    index.add(embeddings)
    return embeddings

# Fonction pour rechercher les documents pertinents
def search_documents(query, k=3):
    query_embedding = model.encode([query])
    D, I = index.search(query_embedding, k)
    return I[0]

# Fonction pour générer une réponse avec RAG
def generate_text_with_rag(prompt, relevant_docs):
    context = "\n".join(relevant_docs)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=[
                {"role": "system", "content": "Vous êtes un assistant utile. Utilisez le contexte suivant pour répondre à la question de l'utilisateur."},
                {"role": "user", "content": f"Contexte: {context}\n\nQuestion: {prompt}"}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Une erreur s'est produite : {e}"

# Fonction pour générer une réponse sans RAG
def generate_text_without_rag(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-2024-05-13",
            messages=[
                {"role": "system", "content": "Vous êtes un assistant utile."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Une erreur s'est produite : {e}"

# Fonction pour extraire le texte d'un fichier PDF
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Fonction pour colorer le code Python
def highlight_code(code):
    formatter = HtmlFormatter(style="friendly")
    highlighted_code = highlight(code, PythonLexer(), formatter)
    return f"<style>{formatter.get_style_defs()}</style>{highlighted_code}"

# Interface Streamlit
st.title("GPT-4 Azure OpenAI Demo avec RAG et sans RAG")

# Zone pour ajouter des documents
st.header("Ajouter des documents")
uploaded_files = st.file_uploader("Téléchargez des fichiers texte ou PDF", accept_multiple_files=True, type=["txt", "pdf"])

if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            content = extract_text_from_pdf(uploaded_file)
        else:
            content = uploaded_file.read().decode("utf-8")
        documents.append(content)
    add_documents(documents)
    st.success(f"{len(documents)} document(s) ajouté(s) à la base de connaissances.")

# Zone de saisie pour le prompt avec RAG
st.header("Générer une réponse avec RAG")
prompt_rag = st.text_area("Entrez votre question ici (avec RAG):", height=150)

# Bouton pour générer la réponse avec RAG
if st.button("Générer une réponse avec RAG"):
    if prompt_rag:
        with st.spinner("Recherche des documents pertinents et génération de la réponse..."):
            relevant_doc_indices = search_documents(prompt_rag)
            relevant_docs = [documents[i] for i in relevant_doc_indices if i < len(documents)]
            response = generate_text_with_rag(prompt_rag, relevant_docs)
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0]
            highlighted_code = highlight_code(code)
            st.markdown(highlighted_code, unsafe_allow_html=True)
        else:
            st.text_area("Réponse:", value=response, height=400)
        st.subheader("Documents pertinents utilisés:")
        for doc in relevant_docs:
            st.write(doc)
    else:
        st.warning("Veuillez entrer une question.")

# Zone de saisie pour le prompt sans RAG
st.header("Générer une réponse sans RAG")
prompt_no_rag = st.text_area("Entrez votre question ici (sans RAG):", height=150)

# Bouton pour générer la réponse sans RAG
if st.button("Générer une réponse sans RAG"):
    if prompt_no_rag:
        with st.spinner("Génération de la réponse..."):
            response = generate_text_without_rag(prompt_no_rag)
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0]
            highlighted_code = highlight_code(code)
            st.markdown(highlighted_code, unsafe_allow_html=True)
        else:
            st.text_area("Réponse:", value=response, height=400)
    else:
        st.warning("Veuillez entrer une question.")

# Informations supplémentaires
st.sidebar.header("À propos")
st.sidebar.info("Cette application utilise l'API Azure OpenAI et RAG pour générer des réponses à vos questions en se basant sur une base de connaissances.")
