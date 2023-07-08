import json
import ir_datasets
from enum import Enum


class TypeDataset(Enum):
    Cranfield = 1
    Trec_covid = 2


class Query:
    def __init__(self, id, text):
        self.id = id
        self.text = text.replace("\n", " ")

    def to_json(self):
        return json.dumps(self.__dict__)


class Qrel:
    def __init__(self, query_id, document_id, relevance):
        self.query_id = str(query_id)
        self.document_id = document_id
        self.relevance = relevance

    def to_json(self):
        return json.dumps(self.__dict__)


class Document:
    def __init__(self, id, title, text):
        self.id = id
        self.title = title.replace("\n", " ")
        self.text = text.replace("\n", " ")

    def to_json(self):
        return json.dumps(self.__dict__)


def create_docs(type_dataset):
    dataset = ir_datasets.load("cord19/trec-covid")
    file = open("trec_covid/docs.txt", "w")

    count = 0
    for doc in dataset.docs_iter():
        if type_dataset == TypeDataset.Cranfield:
            doc_json = Document(doc.doc_id, doc.title.strip(),
                                doc.text.strip()).to_json()
        elif type_dataset == TypeDataset.Trec_covid:
            doc_json = Document(doc.doc_id, doc.title.strip(),
                                doc.abstract.strip()).to_json()
        file.write(doc_json + "\n")
        count += 1

    file.close()
    return count


def create_queries(type_dataset):
    dataset = ir_datasets.load("cord19/trec-covid")
    file = open("trec_covid/query.txt", "w")
    count = 0
    for query in dataset.queries_iter():
        if type_dataset == TypeDataset.Cranfield:
            doc_json = Query(query.query_id, query.text.strip()).to_json()
        elif type_dataset == TypeDataset.Trec_covid:
            doc_json = Query(query.query_id, query.description.strip()).to_json()

        file.write(doc_json + "\n")
        count += 1

    file.close()
    return count


def create_qrels():
    dataset = ir_datasets.load("cord19/trec-covid")
    count = 0
    for query in dataset.queries_iter():
        file = open("trec_covid/qrels_" + query.query_id + ".txt", "w")

        for qrels in dataset.qrels_iter():
            if query.query_id != qrels.query_id:
                continue
            doc_json = Qrel(qrels.query_id, qrels.doc_id, qrels.relevance).to_json()
            file.write(doc_json + "\n")
            count += 1

        file.close()
    return count


if __name__ == '__main__':
    # total = create_docs(type_dataset=TypeDataset.Trec_covid)
    # print("Total created documents:", total)
    #
    # total = create_queries(type_dataset=TypeDataset.Trec_covid)
    # print("Total created queries:", total)

    total = create_qrels()
    print("Total created qrels:", total)
