import pandas as pd
from datasets.informationretrievaltest import InformationRetrievalTest, Collection, Arch


def els_process():
    els = InformationRetrievalTest(collection_name=Collection.Trec_covid, arch=Arch.ELS)
    els.index_documents(url="http://127.0.0.1:9200/")
    els.retrival_documents()
    pd.DataFrame(els.recall_precision_metrics()).to_json("els_fscore.json")
    pd.DataFrame(els.ncdg_metrics()).to_json("els_ndcg.json")


def sira_process():
    sira = InformationRetrievalTest(collection_name=Collection.Trec_covid, arch=Arch.SIRA)
    sira.index_documents(url="http://127.0.0.1:8080/nir")
    sira.retrival_documents()
    pd.DataFrame(sira.recall_precision_metrics()).to_json("sira_fscore.json")
    pd.DataFrame(sira.ncdg_metrics()).to_json("sira_ndcg.json")


if __name__ == '__main__':
    els_process()
    sira_process()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
