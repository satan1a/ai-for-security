# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def k_means_train():
    """
    K-Means算法的展示
    :return:
    """
    # 生成测试样本
    n_samples = 1500
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)

    # 拟合K-Means聚类函数，进行聚类
    y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X)

    # 可视化聚类结果
    plt.figure(figsize=(12, 12))
    plt.subplot(221)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("K-Means Train")
    plt.show()


# def load_dga(dga_file_path):
#     # x = []
#     data = pd.read_csv(dga_file_path, sep='\t', header=None, skiprows=18,)
#     x = [i[1] for i in data.values]
#     return x


if __name__ == '__main__':
    # dga_file_path = "../../data/data-0x2/360-netlab-dga-head1K.txt"
    # x = load_dga(dga_file_path)
    # CV = CountVectorizer(ngram_range=(2, 2), token_pattern=r"\w", decode_error="ascii", stop_words="english",
    #                      max_df=1.0, min_df=1)
    # x = CV.fit_transform(x)
    # print(type(x))

    k_means_train()