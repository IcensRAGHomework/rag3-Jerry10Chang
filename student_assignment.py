import datetime
import chromadb
import traceback
import pandas as pd
import time

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"

def generate_hw01():
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )

    if collection.count() == 0:
        df = pd.read_csv("COA_OpenData.csv")
        for _, row in df.iterrows():
            try:
                metadata = {
                    "file_name": "COA_OpenData.csv",
                    "name": row["Name"],
                    "type": row["Type"],
                    "address": row["Address"],
                    "tel": row["Tel"],
                    "city": row["City"],
                    "town": row["Town"],
                    "date": int(datetime.datetime.strptime(row["CreateDate"], "%Y-%m-%d").timestamp())
                }

                id = row.get("ID", "")
                document = row.get("HostWords", "")
                collection.add(ids=[id], documents=[document], metadatas=[metadata])

            except Exception as e:
                print(f"Error inserting row: {row}")
                traceback.print_exc()

    return collection


    
def generate_hw02(question, city, store_type, start_date, end_date):
    collection = generate_hw01()
    
    query_results = collection.query(
        query_texts=[question],
        n_results=10, 
        where={
            "$and": [
                {"date": {"$gte": int(start_date.timestamp())}},
                {"date": {"$lte": int(end_date.timestamp())}},
                {"type": {"$in": store_type}},
                {"city": {"$in": city}}
            ]
        }
    )

    # print(query_results)
    # for result in query_results:
    #     print(result)

    metadatas = query_results['metadatas'][0]
    distances = query_results['distances'][0]
    match_list = []

    for metadata, distance in zip(metadatas, distances):
        similarity = 1 - distance
        if similarity >= 0.8:
            match_list.append([metadata['name'], similarity])
    
    match_list = sorted(match_list, key=lambda x: x[1], reverse=True)
    final_match_list = [metadata for metadata, similarity in match_list]

    return final_match_list



def generate_hw03(question, store_name, new_store_name, city, store_type):
    collection = generate_hw01()

    # 1. new store name
    store_results = collection.query(
        query_texts=[store_name],
        n_results=10,
        where={"name": {"$eq": store_name}}
    )

    store_ids = store_results['ids'][0]
    store_metadatas = store_results['metadatas'][0]

    for store_id, store_metadata in zip(store_ids, store_metadatas):
        store_metadata['new_store_name'] = new_store_name
        collection.update(ids=[store_id], metadatas=[store_metadata])
    
    # 2. search store
    query_results = collection.query(
        query_texts=[question],
        n_results=10,
        where={
            "$and": [
                {"type": {"$in": store_type}},
                {"city": {"$in": city}}
            ]
        }
    )

    metadatas = query_results['metadatas'][0]
    distances = query_results['distances'][0]
    match_list = []

    for metadata, distance in zip(metadatas, distances):
        similarity = 1 - distance
        if similarity >= 0.8:
            match_list.append([metadata.get("new_store_name", metadata['name']), similarity])
    
    match_list = sorted(match_list, key=lambda x: x[1], reverse=True)
    final_match_list = [metadata for metadata, similarity in match_list]

    return final_match_list

def demo(question):
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    
    return collection

# print(
#     generate_hw02(
#         question="我想要找有關茶餐點的店家",
#         city=["宜蘭縣", "新北市"],
#         store_type=["美食"],
#         start_date=datetime.datetime(2024, 4, 1),
#         end_date=datetime.datetime(2024, 5, 1),
#     )
# )

# print(
#     generate_hw03(
#         question="我想要找南投縣的田媽媽餐廳，招牌是蕎麥麵",
#         store_name="耄饕客棧",
#         new_store_name="田媽媽（耄饕客棧）",
#         city=["南投縣"],
#         store_type=["美食"],
#     )
# )