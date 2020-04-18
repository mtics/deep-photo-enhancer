from pyecharts.charts import *
from pyecharts import options as opts


def data_visualization(file_path, save_path):
    i_losses = list()
    c_losses = list()
    g_losses = list()
    d_losses = list()
    ads = list()
    ags = list()
    gps = list()

    with open(file_path) as file:
        for line in file:
            datas = line.split("] [")
            for data in datas:
                if ("Epoch" not in data) and ("Batch" not in data):

                    name = data.split(":")[0].strip()
                    value = float(data.split(":")[1].strip().replace("]", ""))

                    if name == "I loss":
                        # if value > 0.75:
                        #     value = 0.75
                        i_losses.append(value)
                    elif name == "C loss":
                        # if value > 0.08:
                        #     value = 0.08
                        c_losses.append(value)
                    elif name == "G loss":
                        # if value > 5000:
                        #     value = 5000
                        g_losses.append(value)
                    elif name == "D loss":
                        d_losses.append(value)
                    elif name == "AD":
                        ads.append(value)
                    elif name == "AG":
                        ags.append(value)
                    elif name == "GP":
                        gps.append(value)

    x_list = list()
    for i in range(0, 601):
        x_list.append(i)

    line = Line()
    line.add_xaxis(x_list)
    line.add_yaxis("C Loss", c_losses)
    line.render(save_path+"data_c.html")

    line = Line()
    line.add_xaxis(x_list)
    line.add_yaxis("I Loss", i_losses)
    line.render(save_path+"data_I.html")

    line = Line()
    line.add_xaxis(x_list)
    line.add_yaxis("G Loss", g_losses)
    line.render(save_path+"data_G.html")

    line = Line()
    line.add_xaxis(x_list)
    line.add_yaxis("D Loss", d_losses)
    line.render(save_path+"data_D.html")

    line = Line()
    line.add_xaxis(x_list)
    line.add_yaxis("AD", ads)
    line.render(save_path+"data_AD.html")

    line = Line()
    line.add_xaxis(x_list)
    line.add_yaxis("AG", ags)
    line.render(save_path+"data_AG.html")

    line = Line()
    line.add_xaxis(x_list)
    line.add_yaxis("GP", gps)
    line.render(save_path+"data_GP.html")
