import argparse
import os.path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from tqdm import tqdm

from utils import ROOT

parser = argparse.ArgumentParser()
parser.add_argument('--mode', default='ED', type=str, help='Hemo or ED')
opt = parser.parse_args()

MODE = opt.mode

DATA_DIR_PATH = ROOT / 'data'
OUTPUT_DIR = ROOT / f'logs/2b'

CLUSTERS = 5


# y = a * x^(1/2) * e^(-bx)
def func(x, _a, _b):
    # return _a * x ** 3 + _b * x ** 2
    return _a * np.sqrt(x) * pow(np.e, -_b * x)


def get_data(path):
    table_interval = pd.read_excel(
        path,
        sheet_name='Sheet1',
        header=0,
        usecols=[2, 5, 8, 11, 14, 17, 20, 23, 26],
    )

    table_volume = pd.read_excel(
        path,
        sheet_name='Sheet1',
        header=0,
        usecols=[3, 6, 9, 12, 15, 18, 21, 24, 27],
    )

    intervals_list = []
    volumes_list = []

    for i, row in table_interval.iterrows():

        intervals = []
        volumes = []

        for j, interval in enumerate(row):

            # nan
            if interval != table_interval.iloc[i, j]:
                break

            volume = table_volume.iloc[i, j]

            intervals.append(interval)
            volumes.append(volume)

        intervals_list.append(np.array(intervals))
        volumes_list.append(np.array(volumes))

    return np.array(intervals_list), np.array(volumes_list)


def group_data(x_data, y_data, labels):

    index_groups = []

    for i in range(CLUSTERS):

        index_groups.append(
            [_ for _, value in enumerate(labels) if value == i]
        )

    intervals_list = []
    volumes_list = []

    for index_group in index_groups:

        points = []

        for index in index_group:

            x_axis, y_axis = x_data[index], y_data[index]

            for i in range(len(x_axis)):

                points.append((x_axis[i], y_axis[i]))

        points.sort(key=lambda x: x[0])

        intervals = []
        volumes = []

        last_interval = -1.
        adjusted_interval = 0.

        for point in points:

            interval = point[0]
            volume = point[1]

            if interval - last_interval < 1e-6:

                adjusted_interval += 1e-5
                intervals.append(adjusted_interval)

            else:

                last_interval = interval
                adjusted_interval = interval
                intervals.append(interval)

            volumes.append(volume)

        intervals_list.append(np.array(intervals))
        volumes_list.append(np.array(volumes))

    return np.array(intervals_list), np.array(volumes_list)


def average_kmeans(samples):

    kmeans = KMeans(n_clusters=CLUSTERS, init='k-means++', random_state=0)

    distance_matrix = []

    for iter in tqdm(range(len(samples))):

        params1 = samples[iter]['params']
        distance_vector = []

        for j in range(len(samples)):

            if iter == j:
                distance_vector.append(0.)

            elif iter > j:  # lower triangle
                distance_vector.append(distance_matrix[j][iter])

            else:
                params2 = samples[j]['params']
                distance_vector.append(curve_similarity(params1, params2))

        distance_matrix.append(distance_vector)

    distance_matrix = np.array(distance_matrix)

    # 25 curves per group

    max_iterations = 200
    desired_cluster_size = len(samples) / CLUSTERS

    for iter in range(max_iterations):

        kmeans.fit(distance_matrix)

        labels = kmeans.labels_

        cluster_sizes = [np.sum(labels == i) for i in range(kmeans.n_clusters)]

        oversize_clusters = [i for i, size in enumerate(cluster_sizes) if size > desired_cluster_size]
        undersize_clusters = [i for i, size in enumerate(cluster_sizes) if size < desired_cluster_size]

        if not oversize_clusters or not undersize_clusters:
            break

        for oversize_cluster in oversize_clusters:

            for undersize_cluster in undersize_clusters:

                if cluster_sizes[oversize_cluster] > desired_cluster_size > cluster_sizes[
                    undersize_cluster]:

                    oversize_indices = np.where(labels == oversize_cluster)[0]
                    random_index = np.random.choice(oversize_indices)
                    labels[random_index] = undersize_cluster
                    cluster_sizes[oversize_cluster] -= 1
                    cluster_sizes[undersize_cluster] += 1

    return kmeans.labels_


def curve_similarity(params1, params2):

    x_max, y_max = 2e3, 1e5
    residual_n = 0.

    for x in range(int(x_max)):

        y1, y2 = func(x, *params1), func(x, *params2)
        residual_n += (abs(y1 - y2) / y_max)

    return residual_n


if __name__ == '__main__':

    x_data, y_data = get_data(DATA_DIR_PATH / f'2b{MODE}.xlsx')

    samples = []
    # fit
    for i in range(len(x_data)):

        sample = {'sub_id': 'sub' + str(i).zfill(3)}

        x_axis, y_axis = x_data[i], y_data[i]
        params, *covariance = curve_fit(f=func, xdata=x_axis, ydata=y_axis)

        print(f"params: {params}")

        sample['params'] = params

        samples.append(sample)

    labels = average_kmeans(samples)

    grouped_x_data, grouped_y_data = group_data(x_data, y_data, labels)

    grouped_params = []

    for i in range(len(grouped_x_data)):

        x_axis, y_axis = grouped_x_data[i], grouped_y_data[i]
        params, *covariance = curve_fit(f=func, xdata=x_axis, ydata=y_axis)

        print(f"params: {params}")

        grouped_params.append(params)

        plt.scatter(x_axis, y_axis, s=10, label="data")
        plt.plot(x_axis, func(x_axis, *params), color='red', label="curve")
        plt.xlabel("time")
        plt.ylabel("volume")
        plt.legend()
        plt.savefig(OUTPUT_DIR / f'2b{MODE}{i}.png')
        plt.clf()
        # plt.show()

    for i in range(len(x_data)):

        x_axis, y_axis = x_data[i], y_data[i]
        params = grouped_params[labels[i]]
        total_residual = 0.

        samples[i]['group'] = labels[i]
        samples[i]['params'] = params

        for j in range(len(x_axis)):

            x, y = x_axis[j], y_axis[j]

            pred_y = func(x, *params)
            residual = abs(pred_y - y)

            total_residual += residual

        samples[i]['residual'] = total_residual

    if not os.path.exists(OUTPUT_DIR.parent):
        os.mkdir(OUTPUT_DIR.parent)

    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    keys = list(samples[0].keys())
    output_table = pd.DataFrame([[sample[key] for key in keys] for sample in samples], columns=keys)
    output_table.to_excel(OUTPUT_DIR / f'2b{MODE}result.xlsx', index=False)
