from dotenv import load_dotenv
import random
import json
import os
from openai import AzureOpenAI
from langchain.vectorstores import FAISS
import numpy as np
from tqdm import tqdm
load_dotenv(".env.shared")
load_dotenv(".env.secret")

openai_client = AzureOpenAI(
    api_version=os.environ["OPENAI_API_VERSION"],
    azure_endpoint=os.environ["OPENAI_API_BASE"],
    api_key=os.environ["OPENAI_API_KEY"]
)
db = FAISS.load_local("data/embeddings",openai_client.embeddings.create(model=os.environ["ADA002_DEPLOYMENT"],input=""))

def compute_embeddings(input) :
    embedded_question = openai_client.embeddings.create(input=input, 
                                    model=os.environ["ADA002_DEPLOYMENT"])
    embedded_question = np.array(embedded_question.data[0].embedding).reshape(1, -1)

    return embedded_question


def main() :
    results_list = []
    # query = "I'm looking for examples of End of Well Reports from Vermillion Energy?"
    # query = compute_embeddings(query)
    db = FAISS.load_local("data/embeddings",openai_client.embeddings.create(model=os.environ["ADA002_DEPLOYMENT"],input=""))
    
    # docs = db.similarity_search_by_vector(query[0])


    dict = db.docstore._dict
    list_dict = list(dict.items())

    # Number of random items you want to process
    k = 1500  # Change this to your desired number
    x = 5

    # Select k random items from the dictionary
    random_items = random.sample(list(dict.items()), k)
    
    for count, (key, value) in enumerate(tqdm(random_items, desc="Processing", total=k)):
        query = compute_embeddings(value.page_content)
        try:
            # Attempting the similarity search
            docs = db.similarity_search_with_score_by_vector(query[0], k=5)
        except Exception as e:
            print(f"An error occurred during similarity search: {e}")
            # Optionally, continue to the next iteration or handle the error as needed
            continue
        for result in docs:
            doc = result[0]
            distance = float(result[1])
            index = [idx for idx, key in enumerate(list_dict) if key[1] == doc][0]
            results_list.append({
                "key": key,
                "distance": distance,
                "index": index
            })

        if (count + 1) % x == 0 or count + 1 == k:
            # timestamp = int(time.time())  # Current timestamp
            file_name = f'result_{count}.json'  # File name with timestamp
            with open(file_name, 'w') as file:
                json.dump(results_list, file, indent=4)
            results_list = []  # Reset the list after writing
    print('Done')

if __name__ == "__main__" :
    main()