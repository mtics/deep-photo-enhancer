from pyecharts.charts import *
from pyecharts import options as opts
import numpy as np
from collections import defaultdict


def data_visualization(file_path, save_path):
    i_losses = list()
    c_losses = list()
    g_losses = list()
    d_losses = list()
    ads = list()
    ags = list()
    gps = list()
    g1_losses = list()
    g2_losses = list()

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
                    elif name == "G1 loss":
                        if value > 0.001:
                            value = 0.001
                        g1_losses.append(value)
                    elif name == "G2 loss":
                        if value > 0.001:
                            value = 0.001
                        g2_losses.append(value)

    x_list = list()
    for i in range(0, 150*113):
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

    line = Line()
    line.add_xaxis(x_list)
    line.add_yaxis("G1", g1_losses)
    line.render(save_path+"data_G1.html")

    line = Line()
    line.add_xaxis(x_list)
    line.add_yaxis("G2", g2_losses)
    line.render(save_path+"data_G2.html")



def data_visualization_scatter(file_path, save_path):

    i_losses = list()
    c_losses = list()
    g_losses = list()
    d_losses = list()
    ads = list()
    ags = list()
    gps = list()

    i_dict = defaultdict(list)
    c_dict = defaultdict(list)
    g_dict = defaultdict(list)
    d_dict = defaultdict(list)
    ad_dict = defaultdict(list)
    ag_dict = defaultdict(list)
    gp_dict = defaultdict(list)

    epoch = 0

    with open(file_path) as file:
        for line in file:
            datas = line.split("] [")
            for data in datas:
                if "Batch" not in data:

                    name = data.split(":")[0].strip()
                    if "Epoch" not in data:
                        value = float(data.split(":")[1].strip().replace("]", ""))
                    else:
                        epoch = str(data.split(" ")[1].split("/")[0])

                    if name == "I loss":
                        # if value > 0.75:
                        #     value = 0.75
                        i_dict[epoch].append(value)
                    elif name == "C loss":
                        # if value > 0.08:
                        #     value = 0.08
                        c_dict[epoch].append(value)
                    elif name == "G loss":
                        # if value > 5000:
                        #     value = 5000
                        g_dict[epoch].append(value)
                    elif name == "D loss":
                        d_dict[epoch].append(value)
                    elif name == "AD":
                        ad_dict[epoch].append(value)
                    elif name == "AG":
                        ag_dict[epoch].append(value)
                    elif name == "GP":
                        gp_dict[epoch].append(value)

    scatter = Scatter()
    scatter.add_xaxis(i_dict.keys())
    for key, value_list in i_dict.items():
        scatter.add(key, value_list)
    scatter.render(save_path+"data_i.html")

    scatter = Scatter()
    for key, value_list in c_dict.items():
        nparray = np.array(value_list)
        scatter.add_xaxis(key)
        scatter.add_yaxis(key, value_list)
    scatter.render(save_path+"data_c.html")

    scatter = Scatter()
    for key, value_list in d_dict.items():
        nparray = np.array(value_list)
        scatter.add_xaxis(key)
        scatter.add_yaxis(key, value_list)
    scatter.render(save_path+"data_d.html")

    scatter = Scatter()
    for key, value_list in g_dict.items():
        nparray = np.array(value_list)
        scatter.add_xaxis(key)
        scatter.add_yaxis(key, value_list)
    scatter.render(save_path+"data_g.html")

    scatter = Scatter()
    for key, value_list in ad_dict.items():
        nparray = np.array(value_list)
        scatter.add_xaxis(key)
        scatter.add_yaxis(key, value_list)
    scatter.render(save_path+"data_ad.html")

    scatter = Scatter()
    for key, value_list in ag_dict.items():
        nparray = np.array(value_list)
        scatter.add_xaxis(key)
        scatter.add_yaxis(key, value_list)
    scatter.render(save_path+"data_ag.html")

    scatter = Scatter()
    for key, value_list in gp_dict.items():
        nparray = np.array(value_list)
        scatter.add_xaxis(key)
        scatter.add_yaxis(key, value_list)
    scatter.render(save_path+"data_gp.html")
