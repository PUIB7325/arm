import os
from openai import AzureOpenAI

# Configuration du client Azure OpenAI
client = AzureOpenAI(
    api_key="8295b751c873498280fbbacf6198be97",  
    api_version="2023-03-15-preview",
    azure_endpoint="https://open-ai-dai3-nepal-weu.openai.azure.com"
)

# Fonction pour envoyer une requête à l'API
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
        print(f"Une erreur s'est produite : {e}")
        return None

# Exemple d'utilisation
prompt = "Traduisez 'Bonjour le monde' en anglais :"
result = generate_text(prompt)

if result:
    print(f"Prompt : {prompt}")
    print(f"Réponse : {result}")