import json

import numpy as np

from datasets.main import Query


def qrel_document_id(file_name):
    ids = []
    file = open(file_name, "r")
    for line in file:
        qrel_dic = json.loads(line)
        ids.append(qrel_dic["document_id"])
    return ids


def document_ids(query_id, search_results):
    for result in search_results:
        if result['id'] == query_id:
            return result['documents']
    return []


def f1_score(precision, recall):
    if precision + recall != 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0
    return f1


def true_relevance(file_name):
    relevance = []
    file = open(file_name, "r")
    for line in file:
        qrel_dic = json.loads(line)
        relevance.append(qrel_dic["relevance"])

    return sorted(relevance, reverse=True)


def relevance_document(collection_name, query_id, document_id):
    file = open("datasets/" + collection_name + "/qrels_" + query_id + ".txt", "r")
    for line in file:
        qrel_dic = json.loads(line)
        if qrel_dic["document_id"] == document_id:
            return qrel_dic["relevance"]

    return 0


def relevant_retrieved_docs(collection_name, query_id, documents):
    relevance = []
    for document_id in documents:
        relevance.append(relevance_document(collection_name, query_id, document_id))

    return relevance


def join_retrieved_docs(file_name, collection_name, search_results):
    relevant_retrieved = {}

    file = open(file_name, "r")
    for line in file:
        query_dic = json.loads(line)
        query = Query(query_dic["id"], query_dic["text"])
        retrieved_docs = document_ids(query_id=query.id, search_results=search_results)
        relevant_docs = qrel_document_id(file_name="datasets/" + collection_name + "/qrels_" + query.id + ".txt")
        relevant_retrieved_docs = set(relevant_docs).intersection(set(retrieved_docs))
        if len(relevant_retrieved_docs) > 0:
            relevant_retrieved[query.id] = list(relevant_retrieved_docs)

    return relevant_retrieved


def dcg_at_k(r, k):
    """Calcula o DCG nos primeiros k documentos.
    Args:
    r: lista de relevâncias dos documentos.
    k: número de documentos a considerar.
    Returns:
    DCG nos primeiros k documentos.
    """
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.


def resize_array(array, new_size):
    if new_size < 0:
        raise ValueError("O novo tamanho deve ser maior ou igual a 0.")

    resized_array = array.copy()
    if new_size < len(resized_array):
        resized_array = resized_array[:new_size]
    elif new_size > len(resized_array):
        resized_array.extend([0] * (new_size - len(resized_array)))

    return resized_array