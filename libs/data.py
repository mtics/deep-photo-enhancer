from pyecharts.charts import *
from pyecharts import options as opts


def data_visualization(file_path):
    i_losses = list()
    c_losses = list()
    g_losses = list()
    d_losses = list()

    with open(file_path) as file:
        for line in file:
            datas = line.split("]  [")
            for data in datas:
                if ("Epoch" in data) or ("Batch" in data):
                    break

                name = data.split(":")[0].strip()
                value = float(data.split(":")[1].strip().replace("]", ""))

                if name == "I Loss":
                    i_losses.append(value)
                elif name == "C Loss":
                    c_losses.append(value)
                elif name == "G Loss":
                    g_losses.append(value)
                elif name == "D Loss":
                    d_losses.append(value)

    line = Line("Data Visualization")
    line.add_yaxis("I Loss", i_losses)
    line.add_yaxis("C Loss", c_losses)
    line.add_yaxis("G Loss", g_losses)
    line.add_yaxis("D Loss", d_losses)

    line.show()
