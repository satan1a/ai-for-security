"""
Author：Toky
Function： 机器学习环境测试
"""

import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # 导入糖尿病数据（sklearn自带）
    diabetes = datasets.load_diabetes()
    """
    展示数据集基本信息
    """
    # # 展示数据集的行数和特征维度数量
    # print(diabetes.data.shape)
    # print(diabetes.feature_names)
    # # 展示数据集描述
    # print(diabetes.DESCR)

    """
    选取所需的特征（本次测试使用个人的BMI指数作为特征）
    """
    diabetes_X = diabetes.data[:, np.newaxis, 3]
    # print(diabetes_X)

    """
    划分数据为训练集和测试集
    """
    # 保留最后20个进行测试
    diabetes_X_train = diabetes_X[:-20]
    diabetes_X_test = diabetes_X[-20:]

    diabetes_y_train = diabetes.target[:-20]
    diabetes_y_test = diabetes.target[-20:]

    """
    训练模型：用训练集数据拟合模型
    测试模型：用测试集数据进行预测
    """
    # 创建线性回归模型
    regr = linear_model.LinearRegression()
    # 使用训练集数据训练模型
    regr.fit(diabetes_X_train, diabetes_y_train)
    # 使用测试集预测标签
    diabetes_y_pred = regr.predict(diabetes_X_test)

    """
    通过均方误差和方差误差的大小来计算拟合度
    """
    mean_squared_error_value = mean_squared_error(diabetes_y_test, diabetes_y_pred)
    print("均方误差为：{0}".format(mean_squared_error_value))
    r2_score_value = r2_score(diabetes_y_test, diabetes_y_pred)
    print("方差误差为：{0}".format(r2_score_value))

    """
    绘制可视化的预测结果
    """
    plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
    plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.show()