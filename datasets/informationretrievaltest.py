import json
import requests
from enum import Enum
from datasets.main import Query
from datasets.utils import document_ids, qrel_document_id, f1_score, join_retrieved_docs, resize_array, \
    relevant_retrieved_docs, true_relevance, dcg_at_k


class Collection(Enum):
    Cranfield = "cranfield"
    Trec_covid = "trec_covid"


class Arch(Enum):
    SIRA = 1
    ELS = 2


class InformationRetrievalTest:
    def __init__(self, collection_name, arch):

        if arch == Arch.SIRA or arch == Arch.ELS:
            self.arch = arch
        else:
            raise Exception("Invalid Arch")

        self.collection_name = collection_name.value
        self.search_results = None
        self.documents_file = "datasets/" + collection_name.value + "/docs.txt"
        self.queries_file = "datasets/" + collection_name.value + "/query.txt"

    def retrival_documents(self):
        self.search_results = []
        try:

            file = open(self.queries_file, "r")
            count = 0

            for line in file:
                query_dic = json.loads(line)
                query = Query(query_dic["id"], query_dic["text"])

                try:

                    print("Id:", query.id, " Text:", query.text)

                    if self.arch == Arch.SIRA:
                        response = requests.get("http://127.0.0.1:8080/nir?query=" + query.text)

                        if response.status_code != 200:
                            raise Exception(response.text)

                        documents = []
                        obj = json.loads(response.text)
                        for r in obj["queryResults"]:
                            documents.append(r["document"]["id"])

                        if len(documents) > 0:
                            count += 1
                            self.search_results.append({
                                "id": query.id,
                                "documents": documents
                            })

                    elif self.arch == Arch.ELS:
                        response = requests.get("http://127.0.0.1:9200/" + self.collection_name + "/_doc/_search", json={
                                "query": {
                                    "bool": {
                                        "must": [
                                            {"match": {"title": query.text}},
                                            {"match": {"text": query.text}}
                                        ]
                                    }
                                }
                            })

                        if response.status_code != 200:
                            raise Exception(response.text)

                        obj = json.loads(response.text)
                        docs = obj["hits"]["hits"]
                        documents = []
                        for doc in docs:
                            documents.append(doc["_id"])

                        if len(documents) > 0:
                            count += 1
                            self.search_results.append({
                                "id": query.id,
                                "documents": documents
                            })

                    else:
                        raise Exception("Invalid Arch")

                finally:
                    pass

            file.close()

        finally:
            file.close()

        return self.search_results

    def index_documents(self, url):

        try:
            file = open(self.documents_file, "r")
            count = 0

            for line in file:
                body = json.loads(line)

                try:
                    response = None
                    if self.arch == Arch.SIRA:
                        response = requests.post(url, json=body)
                    elif self.arch == Arch.ELS:
                        response = requests.post(url + self.collection_name + "/_doc/" + body["id"], json=body)
                    else:
                        raise Exception("Invalid Arch")

                    if response.status_code == 201 or response.status_code == 200:
                        count += 1
                    else:
                        print("Status:", response.status_code, "Body:", body)
                        raise Exception(response.text)

                    if count % 1000 == 0:
                        print("Documentos indexados:", count, "Arch:", self.arch)

                finally:
                    pass
                # except Exception as e:
                #     print("Ocorreu um erro:", str(),"Arch:", self.arch)

            print("Documentos indexados:", count, "Arch:", self.arch)
            file.close()

        finally:
            file.close()

    def recall_precision_metrics(self):

        metrics = {
            "query_ids": [],
            "recall": [],
            "precision": [],
            "f_scores": [],
        }

        file = open(self.queries_file, "r")
        for line in file:
            query_dic = json.loads(line)
            query = Query(query_dic["id"], query_dic["text"])
            retrieved_docs = document_ids(query_id=query.id, search_results=self.search_results)
            relevant_docs = qrel_document_id(
                file_name="datasets/" + self.collection_name + "/qrels_" + query.id + ".txt")
            relevant_retrieved_docs = set(relevant_docs).intersection(set(retrieved_docs))

            precision = len(relevant_retrieved_docs) / len(retrieved_docs) if retrieved_docs else 0
            recall = len(relevant_retrieved_docs) / len(relevant_docs) if relevant_docs else 0

            if precision + recall != 0:
                metrics["query_ids"].append(query.id)
                metrics["recall"].append(recall)
                metrics["precision"].append(precision)
                metrics["f_scores"].append(f1_score(precision, recall))

        file.close()
        return metrics

    def ncdg_metrics(self, k=10):
        metrics = {
            "query_ids": [],
            "dcg": [],
            "idcg": [],
            "ndcg": [],
        }

        x = join_retrieved_docs(self.queries_file, self.collection_name, self.search_results)
        for query_id in x:
            relevancies = resize_array(relevant_retrieved_docs(self.collection_name, query_id, x[query_id]), k)
            relevancies_true = resize_array(
                true_relevance(file_name="datasets/" + self.collection_name + "/qrels_" + query_id + ".txt"), k)

            dcg = dcg_at_k(relevancies, k)
            idcg = dcg_at_k(relevancies_true, k)

            metrics["query_ids"].append(query_id)
            metrics["dcg"].append(dcg)
            metrics["idcg"].append(idcg)
            metrics["ndcg"].append(dcg / idcg if idcg != 0 else 0)

        return metrics
