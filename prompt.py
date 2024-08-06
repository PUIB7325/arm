import streamlit as st
from openai import AzureOpenAI

# Configuration du client Azure OpenAI
client = AzureOpenAI(
    api_key="8295b751c873498280fbbacf6198be97",
    api_version="2023-03-15-preview",
    azure_endpoint="https://open-ai-dai3-nepal-weu.openai.azure.com"
)

def generate_text(prompt):
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

# Interface Streamlit
st.title("GPT-4 Azure OpenAI Demo")

# Zone de saisie pour le prompt
prompt = st.text_input("Entrez votre prompt ici:")

# Bouton pour générer la réponse
if st.button("Générer une réponse"):
    if prompt:
        with st.spinner("Génération en cours..."):
            response = generate_text(prompt)
        st.text_area("Réponse:", value=response, height=200)
    else:
        st.warning("Veuillez entrer un prompt.")

# Informations supplémentaires
st.sidebar.header("À propos")
st.sidebar.info("Cette application utilise l'API Azure OpenAI pour générer des réponses à vos prompts.")