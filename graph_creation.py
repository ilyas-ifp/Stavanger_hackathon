import json
import pandas as pd
from utils import concat_json
import networkx as nx


def create_graph() :
    df = concat_json.concat_json()
    df_embed = pd.read_json(r"./json/output.json", lines = True)
    #1 step
    df_embed_keys = df_embed
    df_embed_keys = df_embed_keys.rename(columns = {"Lines": "key" })
    df_embed_keys = pd.merge(df, df_embed_keys, on = "key")
    df_embed_keys = df_embed_keys.rename(columns = {"Embeddings" : "embedding_key"})
    #2 step
    df_embed_index = df_embed
    df_embed_index = df_embed_index.rename(columns = {"Lines": "index" })
    df_embed_index = pd.merge(df, df_embed_index, on = "index")
    df_embed_index = df_embed_index.rename(columns = {"Embeddings" : "embedding_index"})
    df_embed_total = pd.concat([df_embed_keys, df_embed_index["embedding_index"]], axis = 1)
    G = nx.Graph()

    for index, row in df_embed_total.iterrows():
        source_node = row["key"]
        target_node = row["index"]
        embedding_source = row["embedding_key"]
        embedding_target = row["embedding_index"]
        weight = float(row["distance"])
        G.add_edge(source_node, target_node, weight=weight)
        G.add_node(source_node, attribute = embedding_source, id = source_node)
        G.add_node(target_node, attribute = embedding_target, id = target_node)
    print(type(G))
    return G







