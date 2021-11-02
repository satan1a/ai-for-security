# 基于机器学习的恶意URL检测

## 概述

本次实践虽题为”基于机器学习“，但目前也只更新到使用TF-IDF和线性回归模型的进度。不过千里之行始于足下嘛，后面再更啦🐦

## 提出问题

首先我们需要将安全问题进行抽象，也就是针对现状提出问题。在假定的这个业务背景下，我们发现：

-   恶意URL存在特定几种类型1

-   特定类型恶意URL在文本上存在普遍的**词汇特征**，例如钓鱼URL中常见"login", "account", "sigin"等关键词

因此我们尝试使用机器学习算法对恶意URL进行检测分析



## 数据处理

### 样本选择

-   [malicious-URLs](https://github.com/faizann24/Using-machine-learning-to-detect-malicious-URLs)
    -   **malicious-URLs** 在Github上面一个 使用机器学习去检测恶意URL的项目 ，里面有一个训练集，有做标记是正常的URL还是恶意的URL
    -   内容类型：文本样本
    -   是否标记：是
    -   是否特征化：否
    -   使用范围：入侵检测、异常流量、WAF

### 数据清洗

由于样本本身为处理好的标记数据，所以在数据格式和脏数据上无需处理（真实情况下可能正好相反:-(）。

编写数据帧提取函数：

```python
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
```

编写对URL的数据清洗函数：
```python
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
```



## 特征提取

我们首先加载数据集：

```python
    grep_csv_file_path = "../../data/data-0x3/grey-url.csv"
    black_csv_file_path = "../../data/data-0x3/black-url.csv"
    grey_urls, y = csv_data_read(grep_csv_file_path)
```

我们使用TF-IDF算法提取URL的特征，并将数据帧划分为训练集和测试集：

```python
    url_vectorizer = TfidfVectorizer(tokenizer=url_tokenize)
    x = url_vectorizer.fit_transform(grey_urls)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```



## 模型选择

接下来，我们对数据集使用**逻辑回归模型**，并将将拟合后的模型和向量保存为本地文件，便于重复使用

```python
    # 对训练集和测试集执行逻辑回归
    l_regress = LogisticRegression(solver='liblinear')
    l_regress.fit(x_train, y_train)
    l_score = l_regress.score(x_test, y_test)
    # print("测试拟合分数为：{0}".format(l_score))

    file_mode = "../../model/model-0x3/model.pkl"
    dump_model_object(file_mode)
    file_vector = "../../model/model-0x3/vector.pl"
    dump_model_object(file_vector)
```





## 效果评估

在“特征提取”部分我们采用`train_test_split`方法进行随机划分训练集和测试集，进行**交叉验证**，再次回顾代码为：

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

`test_size`：测试集占总样本的占比
`random_state`：随机数的种子

使用线性回归模型，最后得到测试拟合分数为：0.9599966703530615



## 完整代码

```python
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
```



## ToDo

