import numpy as np
import pandas as pd
import datetime as dt
import netCDF4 as nc
from scipy.interpolate import interp1d
import os

dir1 = ''  # the directory where the output trajectory file is saved
dir_app = ""  # the directory where the wind and air density data are located
# mapping dem
# demfilename = 'SakuraDEM.csv'  made from GSI DEM data
dem = pd.DataFrame([[-9999, -9999, -9999, -9999, -9999],
                    [-9999, -9999, 5, 5, -9999],
                    [-9999, 10, 20, 15, 5],
                    [5, 20, 40, 30, 10],
                    [-9999, 5, 10, 5, -9999],
                    [-9999, -9999, -9999, -9999, -9999]])  # pd.read_csv(demfilename, header=0, index_col=0)

range_NS = [0, -1]
range_EW = [0, -1]
time_interval = 20  # minutes
time_range = 180  # minutes
time_slice = time_range // time_interval
z_interval_wind = 200  # m
z_interval_plume = 100  # m
vent_x = 657324  # m UTM
vent_y = 3495137  # m UTM
vent_z = 1000  # m asl

K = 100  # m^2/s
st = dt.datetime.now()

diameter = pd.Series(
    [5.22, 4.42, 4.04, 3.78, 3.58, 3.41, 3.26, 3.13, 3, 2.89, 2.73, 2.53, 2.35, 2.18, 2.01, 1.77, 1.47, 1.19, 0.93,
     0.69, 0.33, -0.06, -0.41, -0.73, -1.01, -1.4, -1.82, -2.19, -2.51, -2.8, -3.17, -3.6],
    index=[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.1, 1.3, 1.5, 1.7, 1.9,
           2.2, 2.6, 3, 3.4, 3.8, 4.4, 5.2, 6, 6.8, 7.6, 8.8, 10.4, 12, 13.6, 15.2, 17.6, 20.8])
F = 0.81 - 0.03 * diameter  # no dimension
cood = pd.DataFrame()
weight = pd.DataFrame()
# loop in each settling velocity class
rho_p = 2640  # kg/m3
eta_a = 1.8e-5  # kg/m s
g = 9.81  # m/s^2
table_det = pd.DataFrame([["2020/2/28 18:00", 1, "m", 1.2, "x", ""],
                         ["2020/2/28 19:15", "", "m", 1, "x", 1]],
                         index=[20201, 20202], columns=["ertime", "site1", "site2", "site3", "site4", "site5"])
# pd.read_csv("detect_list.csv", index_col=0, header=0).fillna(0)
table_er = pd.DataFrame([[155, "Ex", "2020/2/28 18:00", 3000, 8500],
                         ["-", "Er", "2020/2/28 19:15", 1400, 2000]],
                         index=[20201, 20202], columns=["exno", "ex_er", "datetime", "h_p", "ejecta"])
# pd.read_csv("../../13-JMA/eruptlist/JMAexer_190601-210731.csv", index_col=0, header=0).fillna(0)


def time_to_border(u1, u2, d, c1, c2, c3, c4):
    return np.ceil(
        (c1 - (c1 * c2 + c3 * c4) / d * c2) / (u1 - (u1 * c2 + u2 * c4) / d * c2) * 100) / 100


def one_step(u, v, w, x1, x2, x3, x4, y1, y2, y3, y4, Z, d, h, x0, y0, z0, t0, x0n, y0n, z0n, t0n):
    [x01, x13, y01, y13, x12, y12] = [x1 - x0, x3 - x1, y1 - y0, y3 - y1, x2 - x1, y2 - y1]
    [d13, d12] = [x13 ** 2 + y13 ** 2, x12 ** 2 + y12 ** 2]
    xt1 = time_to_border(u, v, d13, x01, x13, y01, y13)
    if xt1 > 0:
        xt2 = time_to_border(v, u, d13, y01, y13, x01, x13)
        if xt1 != xt2:
            print("xt1 != xt2 unchi")
            return np.zeros(19)
        else:
            [xt, x1n] = [xt1, x0n - 1]
    else:
        [x02, x24, y02, y24] = [x2 - x0, x4 - x2, y2 - y0, y4 - y2]
        d24 = x24 ** 2 + y24 ** 2
        xt3 = time_to_border(u, v, d24, x02, x24, y02, y24)
        xt4 = time_to_border(v, u, d24, y02, y24, x02, x24)
        if xt4 != xt3:
            print("d: " + str(d) + "h: " + str(h) + "xt3: " + str(xt3) + " xt4: " + str(xt4) + " unchi")
            return np.zeros(19)
        # elif xt3 == 0:
        #     [xt, x1n] = [float("inf"), x0n]
        # elif xt3 < 0:
        #     if xt1 == 0:
        #         [xt, x1n] = [float("inf"), x0n]
        #     else:
        #         return np.zeros(19)
        else:
            [xt, x1n] = [xt3, x0n + 1]
    yt1 = time_to_border(u, v, d12, x01, x12, y01, y12)
    if yt1 > 0:
        yt2 = time_to_border(v, u, d12, y01, y12, x01, x12)
        if yt1 != yt2:
            print("yt1 != yt2 unchi")
            return np.zeros(19)
        else:
            [yt, y1n] = [yt1, y0n - 1]
    else:
        [x03, x34, y03, y34] = [x3 - x0, x4 - x3, y3 - y0, y4 - y3]
        d34 = x34 ** 2 + y34 ** 2
        yt3 = time_to_border(u, v, d34, x03, x34, y03, y34)
        yt4 = time_to_border(v, u, d34, y03, y34, x03, x34)
        if yt4 != yt3:
            print("d: " + str(d) + "h: " + str(h) + "yt3: " + str(yt3) + " yt4: " + str(yt4) + " unchi")
            return np.zeros(19)
        # elif yt3 == 0:
        #     [yt, y1n] = [float("inf"), y0n]
        # elif yt3 < 0:
        #     if yt1 == 0:
        #         [yt, y1n] = [float("inf"), y0n]
        #     else:
        #         return np.zeros(19)
        else:
            [yt, y1n] = [yt3, y0n + 1]
    if w != 0:
        [zt, z1n] = [np.ceil((Z[int(z0n + (w > 0)), y0n, x0n] - z0) / w * 100) / 100, int(z0n + np.sign(w))]
        if zt == 0:
            zt = float("inf")
    else:
        zt = float('inf')
        z1n = z0n
    return [d, h, x0, y0, z0, t0, x0n, y0n, z0n, t0n, u, v, w, xt, yt, zt, x1n, y1n, z1n]


def rise(erno):
    h_p = table_er.loc[erno, "h_p"]
    sheet_rise = pd.DataFrame(
        columns=["x0", "y0", "z0", "t0", "x0n", "y0n", "z0n", "t0n", "u", "v", "w", "xt", "yt", "zt"])
    ertime = pd.to_datetime(table_det.loc[erno, "ertime"])
    windstart_utc = ertime - dt.timedelta(hours=ertime.hour % 3 + 9, minutes=ertime.minute)  # 9 hours: JST - UTC
    ertime_sec = (ertime - windstart_utc - dt.timedelta(hours=9)).seconds
    winddat = nc.Dataset(dir_app + "" + windstart_utc.strftime("%Y-%m-%d_%H%M%S_intp.nc"), "r")
    ertime_jst_10min = ertime - dt.timedelta(minutes=ertime.minute % time_interval)
    # The time at which the wind data exist just before the eruption start time.
    # Here we use data in 10-minute increments.
    num = int((ertime_jst_10min - windstart_utc - dt.timedelta(hours=9)).seconds / (time_interval * 60))
    U = np.array(winddat.variables["u"])[num:, :, :, :]
    V = np.array(winddat.variables["v"])[num:, :, :, :]
    W = np.array(winddat.variables["w"])[num:, :, :, :]
    X = np.array(winddat.variables["x_utm"])
    Y = np.array(winddat.variables["y_utm"])
    alt = np.array(winddat.variables["alt"])
    Z = np.array([np.array(winddat.variables["topo"]) + alt[i] for i in range(len(alt))])
    if os.path.isfile(dir_app + "" + (windstart_utc + dt.timedelta(hours=3)).strftime("%Y-%m-%d_%H%M%S_intp.nc")):
        winddat2 = nc.Dataset(dir_app + "" + windstart_utc.strftime("%Y-%m-%d_%H%M%S_intp.nc"), "r")
        U = np.concatenate([U, np.array(winddat2.variables["u"])[:]])
        V = np.concatenate([V, np.array(winddat2.variables["v"])[:]])
        W = np.concatenate([W, np.array(winddat2.variables["w"])[:]])
        del winddat2
    sheet_h = pd.DataFrame()
    [x0, y0, z0, t0, x0n, y0n, z0n, t0n] = [vent_x, vent_y, vent_z, ertime_sec, 83, 72,
                                            int((vent_z - Z[0, 72, 83]) // z_interval_wind), 0]
    # 83, 72 is the grid number where the vent is located
    while (z0 < h_p + 1000) & (x0n < X.shape[1] - 1) & (y0n < X.shape[0] - 1) & (x0n > 0) & (y0n > 0) & (
            t0n < len(U) - 1) & (z0n >= 0):
        vt = -1 / 3.2e-5 / 2 / np.max([z0 - 1000, 50])
        [u, v, w] = np.round([U[t0n, z0n, y0n, x0n], V[t0n, z0n, y0n, x0n], W[t0n, z0n, y0n, x0n] - vt], 2)
        [x1, x2, x3, x4, y1, y2, y3, y4] = [X[y0n, x0n], X[y0n, x0n + 1], X[y0n + 1, x0n], X[y0n + 1, x0n + 1],
                                            Y[y0n, x0n], Y[y0n, x0n + 1], Y[y0n + 1, x0n], Y[y0n + 1, x0n + 1]]
        step_set = one_step(u, v, w, x1, x2, x3, x4, y1, y2, y3, y4, Z,
                            0, h_p, x0, y0, z0, t0, x0n, y0n, z0n, t0n)
        if (step_set == np.zeros(19)).all():
            break
        sheet_h = sheet_h.append(pd.DataFrame([step_set[2:-3]], columns=sheet_rise.columns))
        [x0, y0, z0, t0] = np.round(
            step_set[2:6] + np.concatenate((step_set[-9:-6], [1])) * min(
                np.array(step_set[-6:-3])[np.array(step_set[-6:-3]) > 0]), 4)
        [x0n, y0n, z0n] = np.array(
            step_set[6:9] + ([step_set[-3] - x0n, step_set[-2] - y0n, step_set[-1] - z0n]) * (
                    step_set[-6:-3] == min(np.array(step_set[-6:-3])[np.array(step_set[-6:-3]) > 0]))).astype("int")
        if step_set[-4] != min(np.array(step_set[-6:-3])[np.array(step_set[-6:-3]) > 0]):
            z0n = int((z0 - Z[0, y0n, x0n]) // z_interval_wind)
        # the time from when the data wind fields starts [s] divided by time_interval
        t0n = int((t0 - winddat.variables["time"][num] * 60) // (time_interval * 60))
    step_set = one_step(u, v, w, x1, x2, x3, x4, y1, y2, y3, y4, Z, 0, h_p, x0, y0, z0, t0, x0n, y0n, z0n, t0n)
    sheet_h = sheet_h.append(pd.DataFrame([step_set[2:-3]], columns=sheet_rise.columns))
    os.makedirs(dir1 + str(erno) + "/traj" + str(erno) + "/", exist_ok=True)
    sheet_h.to_csv(dir1 + str(erno) + "/traj" + str(erno) + "/rise_K100.csv", index=None)


def traj(erno):
    h_p = table_er.loc[erno, "h_p"]
    shokichi = pd.DataFrame(columns=np.arange(vent_z + z_interval_plume,
                                              h_p + vent_z + z_interval_plume, z_interval_plume), index=diameter.index)
    sheet_rise = pd.read_csv(dir1 + str(erno) + "/traj" + str(erno) + "/rise_K100.csv")
    ertime = pd.to_datetime(table_det.loc[erno, "ertime"])
    sheet = pd.DataFrame(
        columns=["d", "h", "x0", "y0", "z0", "t0", "x0n", "y0n", "z0n", "t0n", "u", "v", "w", "xt", "yt", "zt"])
    windstart_utc = ertime - dt.timedelta(hours=ertime.hour % 3 + 9, minutes=ertime.minute)
    winddat = nc.Dataset(dir_app + "" + windstart_utc.strftime("%Y-%m-%d_%H%M%S_intp.nc"), "r")
    ertime_jst_10min = ertime - dt.timedelta(minutes=ertime.minute % time_interval)  # 10分単位で指定

    # calling wind data from WRF data. creating u(x, y, z, t), v(x, y, z, t), w(x, y, z, t)
    num = int((ertime_jst_10min - windstart_utc - dt.timedelta(hours=9)).seconds / (time_interval * 60))
    U = np.array(winddat.variables["u"])[num:, :, :, :]
    V = np.array(winddat.variables["v"])[num:, :, :, :]
    W = np.array(winddat.variables["w"])[num:, :, :, :]
    rho = np.array(winddat.variables["rho"])[num:, :, :, :]
    if os.path.isfile(dir_app + "" + (windstart_utc + dt.timedelta(hours=3)).strftime("%Y-%m-%d_%H%M%S_intp.nc")):
        winddat2 = nc.Dataset(dir_app + "" + windstart_utc.strftime("%Y-%m-%d_%H%M%S_intp.nc"), "r")
        U = np.concatenate([U, np.array(winddat2.variables["u"])[:]])
        V = np.concatenate([V, np.array(winddat2.variables["v"])[:]])
        W = np.concatenate([W, np.array(winddat2.variables["w"])[:]])
        rho = np.concatenate([rho, np.array(winddat2.variables["rho"])[:]])
        del winddat2
    print(["l35", dt.datetime.now()])

    X = np.array(winddat.variables["x_utm"])
    Y = np.array(winddat.variables["y_utm"])
    alt = np.array(winddat.variables["alt"])
    Z = np.array([np.array(winddat.variables["topo"]) + alt[i] for i in range(len(alt))])
    for d in shokichi.index:
        pocky = rho_p * g * (2 ** (-diameter.loc[d]) / 1000) ** 2
        pletz = 9 * eta_a * F.loc[d] ** -0.32
        toppo = 1.5 * rho_p * g * (2 ** (-diameter.loc[d]) / 1000) ** 3 * (1.07 - F.loc[d]) ** 0.5

        def f_sheet_h(h):
            sheet_h = pd.DataFrame()
            # loop in every segregation height. The departure box is decided depending on the segregation height.
            [x0n, y0n, z0n, t0n] = sheet_rise[sheet_rise["z0"] < h].iloc[-1, 4:8].astype(int)
            z0 = h
            x0 = interp1d(sheet_rise["z0"], sheet_rise["x0"], bounds_error=False)(h)
            y0 = interp1d(sheet_rise["z0"], sheet_rise["y0"], bounds_error=False)(h)
            t0 = interp1d(sheet_rise["z0"], sheet_rise["t0"], bounds_error=False)(h)

            while (z0 - Z[0, y0n, x0n] > 0) & (x0n < X.shape[1] - 1) & (y0n < X.shape[0] - 1) & (x0n > 0) & (
                    y0n > 0) & (t0n < len(U) - 1) & (z0n >= 0) & (z0n < len(Z) - 1):
                vt = np.round(pocky / (pletz + (pletz ** 2 + rho[t0n, z0n, y0n, x0n] * toppo) ** 0.5), 2)
                [u, v, w] = np.round([U[t0n, z0n, y0n, x0n], V[t0n, z0n, y0n, x0n], W[t0n, z0n, y0n, x0n] - vt], 2)
                [x1, x2, x3, x4, y1, y2, y3, y4] = [X[y0n, x0n], X[y0n, x0n + 1], X[y0n + 1, x0n], X[y0n + 1, x0n + 1],
                                                    Y[y0n, x0n], Y[y0n, x0n + 1], Y[y0n + 1, x0n], Y[y0n + 1, x0n + 1]]
                step_set = one_step(u, v, w, x1, x2, x3, x4, y1, y2, y3, y4, Z,
                                    d, h, x0, y0, z0, t0, x0n, y0n, z0n, t0n)
                xt, yt, zt = step_set[-6:-3]
                if (step_set == np.zeros(19)).all():
                    break
                sheet_h = sheet_h.append(pd.DataFrame([step_set[:-3]], columns=sheet.columns))
                x0, y0, z0, t0 = np.round(
                    step_set[2:6] + np.concatenate((step_set[-9:-6], [1])) * min(
                        np.array([xt, yt, zt])[np.array([xt, yt, zt]) > 0]), 4)
                x0n, y0n, z0n = np.array(
                    step_set[6:9] + ([step_set[-3] - x0n, step_set[-2] - y0n, step_set[-1] - z0n]) * (
                            step_set[-6:-3] == min(np.array([xt, yt, zt])[np.array([xt, yt, zt]) > 0]))).astype("int")
                if step_set[-4] != min(np.array([xt, yt, zt])[np.array([xt, yt, zt]) > 0]):
                    z0n = int((z0 - Z[0, y0n, x0n]) // z_interval_wind)
                # the time from when the data wind fields starts [s] divided by time_interval
                try:
                    t0n = int((t0 - winddat.variables["time"][num] * 60) // (time_interval * 60))
                except:
                    break
                # saving calculation time when the particle goes back to the same box as the 2 steps before
                if len(sheet_h) > 2:
                    if ([x0n, y0n, z0n] == sheet_h.iloc[-2, 6:9]).all():
                        vt = np.round(pocky / (pletz + (pletz ** 2 + rho[t0n, z0n, y0n, x0n] * toppo) ** 0.5), 2)
                        [x2n, y2n, z2n] = sheet_h.iloc[-1, 6:9]
                        [u, v, w] = np.where(np.array([x0n, y0n, z0n]) == np.array([x2n, y2n, z2n]),
                                             [np.mean([U[t0n, z2n, y2n, x2n], U[t0n, z0n, y0n, x0n]]),
                                              np.mean([V[t0n, z2n, y2n, x2n], V[t0n, z0n, y0n, x0n]]),
                                              np.mean([W[t0n, z2n, y2n, x2n], W[t0n, z0n, y0n, x0n]]) - vt], 0)
                        [x1, x2, x3, x4, y1, y2, y3, y4] = [X[np.min([y0n, y2n]), np.min([x0n, x2n])],
                                                            X[np.min([y0n, y2n]), np.max([x2n, x0n]) + 1],
                                                            X[np.max([y0n, y2n]) + 1, np.min([x0n, x2n])],
                                                            X[np.max([y0n, y2n]) + 1, np.max([x2n, x0n]) + 1],
                                                            Y[np.min([y0n, y2n]), np.min([x0n, x2n])],
                                                            Y[np.min([y0n, y2n]), np.max([x2n, x0n]) + 1],
                                                            Y[np.max([y0n, y2n]) + 1, np.min([x0n, x2n])],
                                                            Y[np.max([y0n, y2n]) + 1, np.max([x2n, x0n]) + 1]]
                        step_set = one_step(u, v, w, x1, x2, x3, x4, y1, y2, y3, y4, Z,
                                            d, h, x0, y0, z0, t0, x0n, y0n, z0n, t0n)
                        if (step_set == np.zeros(19)).all():
                            break
                        sheet_h = sheet_h.append(pd.DataFrame([step_set[:-3]], columns=sheet.columns))
                        [x0, y0, z0, t0] = np.round(
                            step_set[2:6] + np.concatenate((step_set[-9:-6], [1])) * min(
                                np.array(step_set[-6:-3])[np.array(step_set[-6:-3]) > 0]), 4)
                        [x0n, y0n, z0n] = np.array(
                            step_set[6:9] + ([step_set[-3] - x0n, step_set[-2] - y0n, step_set[-1] - z0n]) * (
                                    step_set[-6:-3] == min(
                                np.array(step_set[-6:-3])[np.array(step_set[-6:-3]) > 0]))).astype("int")
                        if step_set[-4] != min(np.array(step_set[-6:-3])[np.array(step_set[-6:-3]) > 0]):
                            z0n = int((z0 - Z[0, y0n, x0n]) // z_interval_wind)
                        # the time from when the data wind fields starts [s] divided by time_interval
                        try:
                            t0n = int((t0 - winddat.variables["time"][num] * 60) // (time_interval * 60))
                        except:
                            break

            sheet_h = sheet_h.append(pd.DataFrame([[d, h, x0, y0, z0, t0, x0n, y0n, z0n, t0n, u, v, w, xt, yt, zt]],
                                                  columns=sheet.columns))
            return sheet_h

        sheet = sheet.append(pd.concat(list(map(f_sheet_h, shokichi.columns))))
        en = dt.datetime.now()
        if d in shokichi.index[::8]:
            print([d, en - st, len(sheet)])
        if len(sheet) > 10000:
            sheet.to_csv(
                dir1 + str(erno) + "/traj" + str(erno) + "/" + str(int(d * 1000)) + "mms-1_nonrand.csv", index=None)
            sheet = pd.DataFrame(
                columns=["d", "h", "x0", "y0", "z0", "t0", "x0n", "y0n", "z0n", "t0n", "u", "v", "w", "xt", "yt", "zt"])
        elif d == shokichi.index[-1]:
            sheet.to_csv(
                dir1 + str(erno) + "/traj" + str(erno) + "/" + str(int(d * 1000)) + "mms-1_nonrand.csv", index=None)


for event in table_det.index:
    if table_er.loc[event, "h_p"] == 0:
        continue
    traj(event)
