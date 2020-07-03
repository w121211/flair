from flair.data import Corpus
from flair.datasets import CSVClassificationCorpus

# this is the folder in which train, test and dev files reside
data_folder = '/workspace/flair/data/news'

# column format indicating which columns hold the text and label(s)
column_name_map = {0: "text", 1: "label"}

# load corpus containing training, test and dev data and if CSV has a header, you can skip it
corpus = CSVClassificationCorpus(data_folder,
                                         column_name_map,
                                         skip_header=True,
#                                          delimiter='\t',    # tab-separated files
) 
corpus.train[0]