import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import networkx as nx
import matplotlib.pyplot as plt

# Charger le modèle et le tokenizer
model_name = "model_parse_mt5_eval_en+fr-google-quereo2-qald-google"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

def generate_amr(text):
    # Tokeniser le texte
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    # Générer l'AMR
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=512, early_stopping=True)
    # Décoder l'AMR
    amr = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return amr

def parse_amr_to_graph(amr):
    # Cette fonction convertit une chaîne AMR en un graphe NetworkX
    graph = nx.DiGraph()
    lines = amr.split('\n')
    for line in lines:
        if line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) >= 3:
            source = parts[0]
            relation = parts[1]
            target = parts[2]
            graph.add_edge(source, target, label=relation)
    return graph

def draw_graph(graph, title):
    pos = nx.spring_layout(graph)
    edge_labels = nx.get_edge_attributes(graph, 'label')
    nx.draw(graph, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.title(title)
    plt.show()

def compare_amr(original_text, summary_text):
    # Générer les AMR pour les deux textes
    original_amr = generate_amr(original_text)
    summary_amr = generate_amr(summary_text)
    
    # Convertir les AMR en graphes
    original_graph = parse_amr_to_graph(original_amr)
    summary_graph = parse_amr_to_graph(summary_amr)
    
    # Comparer les AMR
    if nx.is_isomorphic(original_graph, summary_graph):
        print("Les AMR sont identiques.")
    else:
        print("Les AMR sont différents.")
    
    # Afficher les graphes
    draw_graph(original_graph, "AMR Original")
    draw_graph(summary_graph, "AMR Résumé")

# Exemple de texte original et résumé
original_text = "Le chat est sur le tapis."
summary_text = "Le chat est sur le tapis."

# Comparer les AMR
compare_amr(original_text, summary_text)
