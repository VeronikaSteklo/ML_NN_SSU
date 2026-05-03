from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def regression_metrics(y_true, y_pred, class_names):
    res = {}
    for i, name in enumerate(class_names):
        res[name] = {
            "MAE": mean_absolute_error(y_true[:, i], y_pred[:, i]),
            "RMSE": np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        }
    res["Global"] = {"MAE": mean_absolute_error(y_true, y_pred),
                     "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))}
    return res
