import os
import shutil
from core.data_utils.build_graph import get_data, process_sentences


if __name__ == "__main__":

    basepath = "./data/sentiment/"
    datasetname ="orig"

    # shutil.rmtree(os.path.join(basepath, datasetname, "processed"))

    x_tr, y_tr, x_ts, y_ts, x_val, y_val, x_all, y_all, adj = get_data("./data/sentiment/", "orig")

