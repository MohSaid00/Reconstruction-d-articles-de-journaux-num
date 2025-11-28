import os
import re
import glob
import numpy as np
import pandas as pd
import networkx as nx
import lightgbm as lgb
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from lxml import etree
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# ================= CONFIGURATION =================
# Tes amis devront changer ce chemin selon leur PC
DATA_FOLDER = "chef-douvre/AS_TrainingSet_BnF_NewsEye_v2"
NS = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

def load_sbert():
    print("--- Chargement du modèle SBERT (MiniLM-L12) ---")
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def get_coords(pts):
    try:
        c = [list(map(int, p.split(','))) for p in pts.split()]
        xs, ys = [i[0] for i in c], [i[1] for i in c]
        return min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)
    except: return 0,0,0,0

def parse_xml(path):
    try:
        tree = etree.parse(path)
        blocks = []
        for r in tree.getroot().findall('.//ns:TextLine', namespaces=NS):
            aid = re.search(r'id:([^;]+)', r.get('custom', ''))
            coords = r.find('ns:Coords', namespaces=NS)
            txt = r.find('.//ns:Unicode', namespaces=NS)
            if aid and coords is not None and txt is not None and txt.text:
                x, y, w, h = get_coords(coords.get('points'))
                # Nettoyage basique
                clean_txt = re.sub(r'[^\w\s]', ' ', txt.text).strip().lower()
                blocks.append({
                    'id': r.get('id'), 'article_id': aid.group(1).strip(),
                    'x': x, 'y': y, 'w': w, 'h': h, 
                    'text': clean_txt,
                    'filename': os.path.basename(path)
                })
        return pd.DataFrame(blocks)
    except: return pd.DataFrame()

def build_features(df, sbert_model, k=8):
    data = []
    for fname, grp in tqdm(df.groupby('filename'), desc="Feature Engineering"):
        if len(grp) < 2: continue
        
        embs = sbert_model.encode(grp['text'].tolist(), convert_to_tensor=True, show_progress_bar=False)
        coords = grp[['x', 'y']].values + grp[['w', 'h']].values/2
        nbrs = NearestNeighbors(n_neighbors=min(len(grp), k+1)).fit(coords)
        _, idxs = nbrs.kneighbors(coords)
        recs = grp.to_dict('records')
        max_h = grp['y'].max()
        
        for i, neighbors in enumerate(idxs):
            for j in neighbors:
                if i == j: continue
                A, B = recs[i], recs[j]
                dy = B['y'] - (A['y'] + A['h'])
                dx = B['x'] - A['x']
                
                # Feature saut de colonne
                is_jump = 1 if (dy < -150 and dx > 0 and A['y'] > max_h*0.5 and B['y'] < max_h*0.5) else 0
                sim = float(util.cos_sim(embs[i], embs[j])[0][0])
                
                data.append({
                    'overlap': (max(0, min(A['x']+A['w'], B['x']+B['w']) - max(A['x'], B['x']))) / (min(A['w'], B['w']) + 1),
                    'dist_y': dy, 'abs_dist_x': abs(dx),
                    'is_jump': is_jump, 'sim': sim, 'jump_score': is_jump * sim,
                    'w_diff': abs(A['w'] - B['w']),
                    'label': 1 if A['article_id'] == B['article_id'] else 0,
                    'filename': fname, 'id_A': A['id'], 'id_B': B['id']
                })
    return pd.DataFrame(data)

def run_pipeline():
    # 1. Lecture
    if not os.path.exists(DATA_FOLDER):
        print(f"ERREUR: Le dossier {DATA_FOLDER} n'existe pas.")
        return

    xml_files = glob.glob(os.path.join(DATA_FOLDER, "*.xml"))[:15] # Test sur 15 fichiers
    print(f"Lecture de {len(xml_files)} fichiers...")
    df_b = pd.concat([parse_xml(f) for f in xml_files])
    
    # 2. Features
    sbert = load_sbert()
    df_l = build_features(df_b, sbert)
    
    # 3. Train
    print("Entraînement LightGBM...")
    cols = ['overlap', 'dist_y', 'abs_dist_x', 'is_jump', 'sim', 'jump_score', 'w_diff']
    # Paramètres optimisés
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=40, max_depth=10, class_weight='balanced', verbose=-1)
    model.fit(df_l[cols], df_l['label'])
    
    # 4. Eval
    df_l['pred'] = (model.predict_proba(df_l[cols])[:, 1] >= 0.5).astype(int)
    scores = []
    results = {}
    
    print("Reconstruction des articles...")
    for fname, links in df_l.groupby('filename'):
        G = nx.Graph()
        valid = links[(links['pred']==1) & ~((links['dist_y']>300) & (links['is_jump']==0))]
        for _, r in valid.iterrows(): G.add_edge(r['id_A'], r['id_B'])
        
        page_nodes = df_b[df_b['filename']==fname]['id'].tolist()
        G.add_nodes_from(page_nodes)
        clusters = [list(c) for c in nx.connected_components(G)]
        
        map_p = {n:i for i,c in enumerate(clusters) for n in c}
        map_t = df_b[df_b['filename']==fname].set_index('id')['article_id'].to_dict()
        yt, yp = [map_t[k] for k in map_t], [map_p.get(k,-1) for k in map_t]
        
        ari = adjusted_rand_score(yt, yp)
        scores.append(ari)
        results[fname] = clusters

    print(f"\n=== RESULTAT FINAL (ARI MOYEN) : {np.mean(scores):.4f} ===")
    
    # Visualisation de la première page
    first_page = list(results.keys())[0]
    print(f"Génération de l'image pour : {first_page}")
    visualiser_resultat(first_page, df_b, results[first_page])

def visualiser_resultat(filename, blocks_df, clusters):
    pb = blocks_df[blocks_df['filename'] == filename]
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_ylim(pb['y'].max()+100, 0); ax.set_xlim(0, pb['x'].max()+pb['w'].max()+100)
    
    cmap = plt.get_cmap('tab20')
    colors = [cmap(i % 20) for i in range(len(clusters))]
    id2col = {n: colors[i] for i,c in enumerate(clusters) for n in c}
    
    for _, b in pb.iterrows():
        c = id2col.get(b['id'], (0.9, 0.9, 0.9, 1))
        ax.add_patch(patches.Rectangle((b['x'],b['y']), b['w'], b['h'], linewidth=1, edgecolor=c, facecolor=c, alpha=0.5))
        
    plt.title(f"Segmentation : {filename} ({len(clusters)} articles)")
    # Sauvegarder l'image au lieu de l'afficher pour un script
    plt.savefig(f"resultat_{filename}.png")
    print(f"Image sauvegardée : resultat_{filename}.png")

if __name__ == "__main__":
    run_pipeline()