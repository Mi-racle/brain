import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd

from utils import ROOT

DATA_PATH = ROOT / 'data'


# 定义要拟合的函数模型，这里使用线性模型 y = mx + b
def linear_model(x, a, b):
    return a * np.sqrt(x) * pow(np.e, -b * x)


def get_data(path):
    table = pd.read_excel(
        path,
        sheet_name='Sheet1',
        header=0,
        usecols=[2, 3],
    )

    intervals = []
    volumes = []

    last_interval = -1.
    adjusted_interval = 0.

    for i, row in table.iterrows():

        interval = table.iloc[i, 0]
        volume = table.iloc[i, 1]

        if interval - last_interval < 1e-6:

            adjusted_interval += 1e-5
            intervals.append(adjusted_interval)

        else:

            last_interval = interval
            adjusted_interval = interval
            intervals.append(interval)

        volumes.append(volume)

    return np.array(intervals), np.array(volumes)


# 生成一些模拟数据
x_data, y_data = get_data(DATA_PATH / '2a.xlsx')

# 利用curve_fit拟合数据
params, covariance = curve_fit(linear_model, x_data, y_data)

# 拟合后的参数
a, b = params

# 打印拟合参数
print("拟合参数:", params)
print(f"Intercept (b): {b}")

# 绘制拟合结果
plt.scatter(x_data, y_data, label="数据")
plt.plot(x_data, linear_model(x_data, a, b), color='red', label="拟合线")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()


