import pandas as pd
import csv

data = pd.read_csv('/media/yassine/Data/PhD/Pytorch/ArabicCQA/embeddings/fasttext.webteb.100d.vec',
                   sep=" ",
                   index_col=0,
                   header=None,
                   skiprows=1,
                   quoting=csv.QUOTE_NONE, encoding='utf8')

print(data.loc['و'])
