from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import networkx as nx
import random

def dijkstra(graph, start):
    # Initialisation
    V = {start}  # L'ensemble V commence avec le nœud source
    distances = {node: float('infinity') for node in graph.nodes}  # d_i = ∞ ∀i ≠ s
    distances[start] = 0  # d_s = 0
    predecessors = {node: None for node in graph.nodes}  # Prédécesseurs
    all_nodes = set(graph.nodes)  # Tous les nœuds du graphe
    steps = []  # Pour stocker l'état à chaque étape
    
    # Itérations: Tant que V ≠ Ø
    while V:
        print("\nV= ", V)
        
        # Sélectionner un nœud i dans V tel que d_i = min_{j∈V} d_j
        min_dist = float('infinity')
        current_node = None
        for node in V:
            if distances[node] < min_dist:
                min_dist = distances[node]
                current_node = node
        
        if current_node is None:
            break  # Plus de nœuds atteignables
        print("current_node= ", current_node)
        # Retirer i de V et enregistrer l'étape
        V_copy = V.copy()  # Sauvegarder V avant suppression pour le tableau
        V.remove(current_node)
        steps.append((current_node, dict(distances), dict(predecessors), V_copy))
        
        # Pour chaque arc (i, j) sortant de i
        for neighbor in graph.neighbors(current_node):
            weight = graph[current_node][neighbor]['weight']
            # Si d_j > d_i + a_ij alors
            if distances[neighbor] > distances[current_node] + weight:
                # d_j ← d_i + a_ij
                distances[neighbor] = distances[current_node] + weight
                predecessors[neighbor] = current_node
                # Ajouter j à V (si non présent)
                if neighbor in all_nodes and neighbor not in V:
                    V.add(neighbor)
    
    return distances, predecessors, steps

def get_path(predecessors, target):
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = predecessors[current]
    return path[::-1]  # Inverser pour obtenir le chemin du départ à la cible

def animate_dijkstra(graph, steps, pos, title, path=None):
    # Créer une figure avec deux sous-graphiques: un pour le graphe, un pour le tableau
    fig = plt.figure(figsize=(10, 8))
    ax_graph = fig.add_subplot(2, 1, 1)  # Graphe en haut
    ax_table = fig.add_subplot(2, 1, 2)  # Tableau en bas
    ax_table.axis('off')  # Masquer les axes du tableau

    # Initialiser le tableau avec les numéros des nœuds comme en-têtes
    num_nodes = len(graph.nodes)
    table_headers = ["Itér", "V"] + [str(node) for node in sorted(graph.nodes)] + ["traiter"]
    table_data = [table_headers] + [["" for _ in table_headers]]
    table = ax_table.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.5, 1.5)

    # Ajuster les largeurs des colonnes du tableau
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
        cell.set_height(0.1)
        if col == 0:  # Colonne Itération
            cell.set_width(0.08)  # Légèrement réduite pour plus de clarté
        elif col == 1:  # Colonne V
            cell.set_width(0.18)  # Largeur augmentée pour la colonne V
        elif col == len(table_headers) - 1:  # Colonne traiter
            cell.set_width(0.12)
        else:  # Colonnes des distances
            cell.set_width(0.06)  # Légèrement réduite pour équilibrer

    def update(frame):
        # Effacer les axes du graphe
        ax_graph.clear()

        # Extraire les données de l'étape
        current_node, distances, predecessors, V = steps[frame]

        # Dessiner le graphe
        nx.draw(graph, pos, with_labels=True, node_color='lightblue', 
                node_size=500, font_size=10, font_weight='bold', ax=ax_graph)
        edge_labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, ax=ax_graph)

        # Mettre en évidence le nœud courant
        nx.draw_networkx_nodes(graph, pos, nodelist=[current_node], node_color='orange', ax=ax_graph)

        # Dessiner l'arbre des plus courts chemins courant
        edges = [(predecessors[n], n) for n in graph.nodes if predecessors[n] is not None]
        nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color='red', width=2, ax=ax_graph)

        # Mettre en évidence le chemin final si fourni et dernière étape
        if path and frame == len(steps) - 1:
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            nx.draw_networkx_edges(graph, pos, edgelist=path_edges, edge_color='green', width=3, ax=ax_graph)

        ax_graph.set_title(f"{title} - Étape {frame+1}/{len(steps)} - target {target}")

        # Mettre à jour le tableau avec les étiquettes numériques des nœuds
        table_row = [
            str(frame + 1),  # Itération comme chaîne pour un affichage cohérent
            str(V),          # V avec étiquettes numériques des nœuds
        ] + [
            distances[node] if distances[node] != float('infinity') else "∞" for node in sorted(graph.nodes)
        ] + [
            current_node     # traiter avec étiquette numérique du nœud
        ]
        table_data[1] = table_row
        for (row, col), cell in table.get_celld().items():
            if row == 0:  # Garder la ligne d'en-tête inchangée
                continue
            cell.set_text_props(text=table_data[row][col])

    # Créer l'animation avec toutes les étapes
    ani = FuncAnimation(fig, update, frames=len(steps), interval=1000, repeat=False)
    ani.save(f"{title.replace(' ', '_')}.gif", writer='pillow', fps=0.5)
    plt.close()

# Définir les graphes et leurs nœuds cibles
graphs = []
# Graphe 1: Petit graphe manuel (6 nœuds)
G1 = nx.DiGraph()  # <- Remplacé nx.Graph() par nx.DiGraph()
edges1 = [(0, 1, 4), (0, 2, 8), (1, 2, 2), (1, 3, 5), (2, 3, 5), (2, 4, 9), (3, 4, 3), (3, 5, 2), (4, 5, 6)]
G1.add_weighted_edges_from(edges1)
graphs.append((G1, "Petit graphe manuel", 5))

# Graphe 2: Graphe en arbre (7 nœuds)
G2 = nx.DiGraph()  # <- Remplacé nx.Graph() par nx.DiGraph()
edges2 = [
    (0, 1, random.randint(1, 10)),
    (1, 2, random.randint(1, 10)),
    (1, 3, random.randint(1, 10)),
    (2, 4, random.randint(1, 10)),
    (3, 5, random.randint(1, 10)),
    (3, 6, random.randint(1, 10))
]
G2.add_weighted_edges_from(edges2)
graphs.append((G2, "Graphe en arbre ", 5))

# Graphe 3: Graphe avec cycles et degrés variés (12 nœuds)
G3 = nx.DiGraph()  # <- Remplacé nx.Graph() par nx.DiGraph()
edges3 = [(0, 1, 3), (0, 2, 6), (1, 3, 2), (2, 3, 4), (2, 4, 7), (3, 5, 1), (4, 5, 3),
          (4, 6, 5), (5, 7, 2), (6, 7, 4), (6, 8, 6), (7, 9, 3), (8, 9, 2), (9, 10, 5),
          (10, 11, 4), (11, 0, 3)]
G3.add_weighted_edges_from(edges3)
graphs.append((G3, "Graphe avec cycles", 9))

# Graphe bipartite
G_bipartite = nx.DiGraph()  # <- Remplacé nx.Graph() par nx.DiGraph()

# Les autres graphes aussi:
G5 = nx.erdos_renyi_graph(6, 0.5, directed=True)  # Graphe aléatoire orienté
for (u, v) in G5.edges():
    G5.edges[u, v]['weight'] = random.randint(1, 10)
graphs.append((G5, "Graphe généré automatiquement", 5))

# Graphe non connecté orienté
G_disconnected = nx.DiGraph()
G_disconnected.add_weighted_edges_from([(0, 1, 3), (1, 2, 5), (2, 3, 2), (3, 0, 4)])
G_disconnected.add_weighted_edges_from([(4, 5, 2), (5, 6, 1), (6, 7, 3), (7, 4, 5)])
graphs.append((G_disconnected, "Graphe_non_connecté", 3))

# Graphe à chemin unique orienté
G_single_path = nx.DiGraph()
edges_single_path = [(i, i+1, random.randint(1, 5)) for i in range(5)]
G_single_path.add_weighted_edges_from(edges_single_path)
graphs.append((G_single_path, "Graphe_chemin_unique", 5))

# Graphe complet orienté
G_complete = nx.complete_graph(6, create_using=nx.DiGraph())
for (u, v) in G_complete.edges():
    G_complete.edges[u, v]['weight'] = random.randint(1, 10)
graphs.append((G_complete, "Graphe_complet", 5))

# Graphe cycle orienté avec poids égaux
G_equal_weights = nx.DiGraph()
edges_cycle = [(i, (i+1) % 8, 1) for i in range(8)]  # Cycle orienté
G_equal_weights.add_weighted_edges_from(edges_cycle)
graphs.append((G_equal_weights, "Graphe_poids_égaux", 7))


# Générer un GIF pour chaque graphe
for graph, title, target in graphs:
    distances, predecessors, steps = dijkstra(graph, 0)
    pos = nx.spring_layout(graph, seed=42)  # Graine fixe pour une disposition cohérente
    if(title=="Graphe bipartite"):
        pos = nx.bipartite_layout(G_bipartite, U)
    path = get_path(predecessors, target)
    animate_dijkstra(graph, steps, pos, title, path)