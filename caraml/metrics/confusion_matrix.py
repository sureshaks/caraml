from numpy import zeros, array, unique

def confusion_matrix(y_true, y_pred):
    n = len(unique(y_true))
    cf = zeros((n, n))
    for i in range(len(y_true)):
        cf[y_true[i][0]][y_pred[i][0]] += 1
    return cf