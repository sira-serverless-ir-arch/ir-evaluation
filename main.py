import json

import pandas as pd
from datasets.informationretrievaltest import InformationRetrievalTest, Collection, Arch


def els_process():
    els = InformationRetrievalTest(collection_name=Collection.Trec_covid, arch=Arch.ELS)
    els.index_documents(url="http://127.0.0.1:9200/")
    els.retrival_documents()
    pd.DataFrame(els.recall_precision_metrics()).to_json("els_trec_covid_fscore.json")
    pd.DataFrame(els.ncdg_metrics()).to_json("els_trec_covid_ndcg.json")


def sira_process():
    sira = InformationRetrievalTest(collection_name=Collection.Trec_covid, arch=Arch.SIRA)
    sira.index_documents(url="http://127.0.0.1:8080/nir")
    sira.retrival_documents()
    pd.DataFrame(sira.recall_precision_metrics()).to_json("test_sira_trec_covid_fscore.json")
    pd.DataFrame(sira.ncdg_metrics()).to_json("test_sira_trec_covid_ndcg.json")

def sira_performance():
    sira = InformationRetrievalTest(collection_name=Collection.Trec_covid, arch=Arch.SIRA)
    # Salvar em um arquivo JSON
    with open("performance/performance_95000.json", "w") as f:
        json.dump(sira.performance(95000), f, indent=4)

if __name__ == '__main__':
    sira_performance()