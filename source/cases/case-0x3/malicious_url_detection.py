"""
Author: Toky
Description: 基于机器学习的恶意URL检测
"""
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def csv_data_read(csv_file_path):
    # 为减少训练时间，可只取头部10W条，但一定需要先打乱样本）
    # df_csv = pd.read_csv(csv_file_path).head(100000)
    df_csv = pd.read_csv(csv_file_path)
    urls = []
    labels = []
    for index, row in df_csv.iterrows():
        urls.append(row["url"])
        labels.append(row["label"])
    return urls, labels


def url_tokenize(url):
    """
    对URL进行清洗，删除斜线、点、和com，进行分词
    :param url:
    :return:
    """
    web_url = url.lower()
    dot_slash = []
    slash = str(web_url).split('/')
    for i in slash:
        r1 = str(i).split('-')
        token_slash = []
        for j in range(0,len(r1)):
            r2 = str(r1[j]).split('.')
            token_slash = token_slash + r2
        dot_slash = dot_slash + r1 + token_slash
    urltoken_list = list(set(dot_slash))
    white_words = ["com", "http:", "https:", ""]
    for white_word in white_words:
        if white_word in urltoken_list:
            urltoken_list.remove(white_word)
    return urltoken_list


def dump_model_object(file_path):
    """
    使用pickle将内存中的对象转换为文本流保存为本地文件
    :param file_path:
    :return:
    """
    with open(file_path, "wb") as f:
        pickle.dump(l_regress, f)
    f.close()


if __name__ == '__main__':
    """
    加载数据集
    """
    grep_csv_file_path = "../../data/data-0x3/grey-url.csv"
    black_csv_file_path = "../../data/data-0x3/black-url.csv"
    grey_urls, y = csv_data_read(grep_csv_file_path)

    """
    使用TF-IDF算法提取关键词特征，并将数据帧划分为训练集和测试集
    """
    url_vectorizer = TfidfVectorizer(tokenizer=url_tokenize)
    x = url_vectorizer.fit_transform(grey_urls)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    """
    对数据帧执行逻辑回归，将拟合后的模型和向量保存
    """
    # 对训练集和测试集执行逻辑回归
    l_regress = LogisticRegression(solver='liblinear')
    l_regress.fit(x_train, y_train)
    l_score = l_regress.score(x_test, y_test)
    print("测试拟合分数为：{0}".format(l_score))

    file_mode = "../../model/model-0x3/model.pkl"
    dump_model_object(file_mode)
    file_vector = "../../model/model-0x3/vector.pl"
    dump_model_object(file_vector)
