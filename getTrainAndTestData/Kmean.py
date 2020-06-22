from sklearn.cluster import KMeans
import numpy as np
# a =[[1],[2], [1], [3]]
# a = np.array(a)
# y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(a)
# print(y_pred)
def getD(data):
    Data = []
    for i in range(data.shape[0]):
        Data.append([int(data[i])])
    Data = np.array(Data)
    if Data.shape[0] > 1:
        y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(Data)
        # print(y_pred)
        zero_index = np.where(y_pred == 0)
        one_index = np.where(y_pred == 1)
        zero_num = zero_index[0].shape[0]
        one_num = one_index[0].shape[0]

        if zero_num > one_num:
            y_pred[zero_index] = 1
            y_pred[one_index] = 0
        # print(y_pred)
        data = data*y_pred
        depth = np.sum(data)//np.sum(y_pred)
    else:
        depth= Data[0]
    return depth
