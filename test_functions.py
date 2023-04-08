import os
import shutil
from core.data_utils.build_graph import get_data


if __name__ == "__main__":
    shutil.rmtree(os.path.join("./data/sentiment/", "orig", "processed"))
    x_tr, y_tr, x_ts, y_ts, x_val, y_val, x_all, y_all, adj = get_data("./data/sentiment/", "orig")