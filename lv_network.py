import pandapower as pp
import pandapower.networks as pn
import random
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import warnings


def make_like_innogy(net):
    """
    change residential subnetwork to innogy standard according to ATMOTERM analysis
    transformer mva is changed and loads are scaled to load transformer to specific percentage
    :param net: pandapower network model
    :return: modified net object
    """
    residentials_loads = net.load['name'].apply(lambda x: 'R' in x)
    koef = calculate_koef()
    typical_loading_percentage = 0.15*koef
    trafo_mva = 0.510
    net.trafo.loc[0, 'sn_mva'] = trafo_mva
    sum_loads = trafo_mva*typical_loading_percentage
    current_loads = net.load['p_mw'][residentials_loads]
    load_scaler = sum_loads/(current_loads.sum())
    net.load['p_mw'][residentials_loads] = net.load['p_mw'][residentials_loads]*load_scaler
    net.line['max_i_ka'] = net.line['max_i_ka']*load_scaler     # let scaling of line nominal powers be same as loads
    return net


def calculate_koef():
    """
    koef is an estimation of how much the 15% average loading of transformer will need to be multiplied by to get peak
    loading of average SN/nn transformer in Poland
    :return: koef
    """
    profile_load = pd.read_excel('Zuzycie energii elektrycznej.xlsm')['G11']
    koef = max(profile_load) / (profile_load.mean())
    return koef


def simulate_scenarios(net_original, n, perc_increase):
    """
    Montecarlo simulation of adding specific amount of additional load to the network
    sizes are defined in p1, p2 and p3, locations are chosen randomly among residential loads
    :param n: (int) number of load distribution scenarios to simulate
    :param perc_increase: (float) total percentage of load increase w.r.t. current load
    :return:
    """
    residentials_loads = net_original.load['name'].apply(lambda x: 'R' in x)
    current_loads = net_original.load['p_mw'][residentials_loads]
    # powers in MW
    p1 = (1.92 / 1000, 0.1)
    p2 = (7 / 1000, 0.9)
    p3 = (70 / 1000, 1)
    max_p = perc_increase * (current_loads.sum())/100
    violation = 0
    df_violations = pd.DataFrame(columns=['n_overtrafos', 'n_overlines', 'n_underbuses'])
    for i in range(n):
        net = deepcopy(net_original)
        res_buses = net.load['bus'][residentials_loads]
        max_index_res_buses = len(res_buses)
        current_p = 0
        while current_p < max_p:
            x = random.uniform(0, 1)
            y = random.uniform(0, 1)
            bus_k = list(res_buses)[int(y * max_index_res_buses)]
            if x < p1[1]:
                load = min(p1[0], max_p - current_p)
                add_load_to_bus_k(net=net, k=bus_k, load=load)
                current_p += load
            elif (x > p1[1]) and (x < p2[1]):
                load = min(p2[0], max_p - current_p)
                add_load_to_bus_k(net=net, k=bus_k, load=load)
                current_p += load
            elif (x > p2[1]) and (x < 1):
                load = min(p3[0], max_p - current_p)
                add_load_to_bus_k(net=net, k=bus_k, load=load)
                current_p += load
            else:
                raise Exception('WTF just happened?')
            # print (current_p)
        pp.runpp(net=net)
        violation += int((net.res_line['loading_percent'] > 100).any())
        n_overlines = sum(net.res_line['loading_percent'] > 100)
        n_overtrafos = sum(net.res_trafo['loading_percent'] > 50)
        n_underbuses = sum(net.res_bus['vm_pu']<0.9)
        df_violations = df_violations.append({'n_overtrafos': n_overtrafos, 'n_overlines': n_overlines,
                                              'n_underbuses': n_underbuses}, ignore_index=True)
        # idea is to get a histogram of how many lines where violated during each perc_increase, and then put multiple
        # historgrams on one plot
    return violation, df_violations


def add_load_to_bus_k(net, k, load):
    index = net.load[net.load['bus'] == k].index[0]
    net.load.loc[index, 'p_mw'] += load


if __name__ == '__main__':
    plt.style.use('ggplot')
    net = pn.create_cigre_network_lv()
    make_like_innogy(net=net)
    list_of_perc = [0, 10, 20, 30, 40, 50, 70, 100, 120, 150]
    df_buses = pd.DataFrame(columns=['n_underbuses'], index=list_of_perc)
    df_lines = pd.DataFrame(columns=['n_overlines'], index=list_of_perc)
    df_trafos = pd.DataFrame(columns=['n_overtrafos'], index=list_of_perc)
    for perc_increase in list_of_perc:
        n_violations, df_violations = simulate_scenarios(net_original=net, n=30, perc_increase=perc_increase)
        # df_nlines[perc_increase] = df_violations['n_overlines']
        # plt.figure()
        # sns.distplot(df_violations['n_underbuses'], label=str(perc_increase), kde=False)
        df_buses.loc[perc_increase] = df_violations['n_underbuses'].mean()
        df_lines.loc[perc_increase] = df_violations['n_overlines'].mean()
        df_trafos.loc[perc_increase] = df_violations['n_overtrafos'].mean()-2

    # df_buses.plot()
    # df_lines.plot()
    # df_trafos.plot()

    sns.lineplot(x=list(df_buses.index), y=list(df_buses['n_underbuses']), label='liczba szyn z zbyt niskim napięciem', markers=True)
    sns.lineplot(x=list(df_lines.index), y=list(df_lines['n_overlines']), label='liczba przeciążonych linii', markers=True)
    sns.lineplot(x=list(df_trafos.index), y=list(df_trafos['n_overtrafos']), label='liczba przeciążonych transformatorów', markers=True)
    plt.title('Uśredniony wpływ dodatkowej mocy na elementy sieci')
    plt.xlabel('dodatkowa moc w systemie, %')
    plt.ylabel('liczba')
    plt.show()