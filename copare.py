import penman
import networkx as nx
import json

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    return content

def amr_to_graph(amr_tree):
    G = nx.DiGraph()
    
    def traverse(node):
        if hasattr(node, 'source'):
            G.add_node(node.source)
            for edge, target in node.attributes():
                G.add_edge(node.source, target, label=edge)
            for edge, target in node.edges():
                G.add_edge(node.source, target.source, label=edge)
                traverse(target)
    
    traverse(amr_tree)
    return G

def create_dataset(ori_file, gpt_file):
    ori_content = read_file(ori_file)
    gpt_content = read_file(gpt_file)
    
    try:
        ori_tree = penman.parse(ori_content)
        ori_graph = amr_to_graph(ori_tree)
    except Exception as e:
        print(f"Erreur lors du traitement du fichier ori.txt: {str(e)}")
        print("Le fichier ori.txt sera traité comme du texte.")
        ori_graph = None
    
    dataset = [{
        "id": 0,
        "conversation_amr": nx.node_link_data(ori_graph) if ori_graph else ori_content.strip(),
        "summary_text": gpt_content.strip()
    }]
    
    return dataset

def save_dataset(dataset, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

# Utilisation du code
ori_file = 'ori.txt'
gpt_file = 'gpt.txt'
output_file = 'dataset_amr.json'

dataset = create_dataset(ori_file, gpt_file)
save_dataset(dataset, output_file)

print(f"Jeu de données créé et sauvegardé dans {output_file}")