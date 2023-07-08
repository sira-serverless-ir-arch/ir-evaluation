from datasets.informationretrievaltest import InformationRetrievalTest, CollectionName


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    collection = InformationRetrievalTest(collection_name=CollectionName.Trec_covid)
    # collection.index_documents()
    collection.retrival_documents()
    collection.recall_precision_metrics()

    r = collection.ncdg_metrics()
    print(r)

    r = collection.recall_precision_metrics()
    print(r)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
