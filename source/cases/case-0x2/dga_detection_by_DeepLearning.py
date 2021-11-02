import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def load_dga(dga_file_path):
    # x = []
    data = pd.read_csv(dga_file_path, sep='\t', header=None, skiprows=18,)
    x = [i[1] for i in data.values]
    return x


if __name__ == '__main__':
    dga_file_path = "../../data/data-0x2/360-netlab-dga-head1K.txt"
    x = load_dga(dga_file_path)
    CV = CountVectorizer(ngram_range=(2, 2), token_pattern=r"\w", decode_error="ascii", stop_words="english",
                         max_df=1.0, min_df=1)
    x = CV.fit_transform(x)
    print(type(x))