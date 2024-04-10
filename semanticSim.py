"""
The following script is a modified version of semantic_search.py by Nils Reimers, available at:
https://github.com/UKPLab/sentence-transformers/blob/master/examples/applications/semantic-search/semantic_search.py

"""
import re
from sentence_transformers import SentenceTransformer, util
import torch

embedder = SentenceTransformer('all-MiniLM-L6-v2')


with open("text_corpus.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    i = 0
    corpus_hash = {}
    while i < len(lines):
        lines[i].strip()
        if re.search(r'^&', lines[i]):
            text_label = lines[i]
            text_body = lines[i+1]
            corpus_hash[text_body] = text_label
            i += 1  
        i += 1
corpus = list(corpus_hash.keys())

print ("calculating embeddings\n")
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)


queries = []    
user_input = input("Please enter your query: ")
queries.append(user_input)


# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(5, len(corpus))
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        print(corpus_hash[corpus[idx]]+" "+corpus[idx], "(Score: {:.4f})".format(score))
