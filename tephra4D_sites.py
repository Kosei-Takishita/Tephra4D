import numpy as np
import pandas as pd
import datetime as dt
from scipy.interpolate import interp1d
from scipy import stats
import glob

# 保存先ディレクトリ
direc2 = ''  # if you want to save or read files in another directory, please set
dir_app = ''  # if you want to save or read files in another directory, please set"
# mapping dem
# demfilename = 'SakuraDEM.csv'  made from GSI DEM data
dem = pd.DataFrame([[-9999, -9999, -9999, -9999, -9999],
                    [-9999, -9999, 5, 5, -9999],
                    [-9999, 10, 20, 15, 5],
                    [5, 20, 40, 30, 10],
                    [-9999, 5, 10, 5, -9999],
                    [-9999, -9999, -9999, -9999, -9999]])  # pd.read_csv(demfilename, header=0, index_col=0)
site = pd.DataFrame(columns=['h', 'd', 'x', 'y'],
                        data=[[10, 500, 400, 300]],
                        index=['site0'])
# [m asl, m from the vent, m to the East from the vent, m to the North from the vent]
table_det = pd.DataFrame([["2020/2/28 18:00", 1, "m", 1.2, "x", ""],
                         ["2020/2/28 19:15", "", "m", 1, "x", 1]],
                         index=[20201, 20202], columns=["ertime", "site1", "site2", "site3", "site4", "site5"])
# pd.read_csv("detect_list.csv", index_col=0, header=0).fillna(0)
table_er = pd.DataFrame([[155, "Ex", "2020/2/28 18:00", 3000, 8500],
                         ["-", "Er", "2020/2/28 19:15", 1400, 2000]],
                         index=[20201, 20202], columns=["exno", "ex_er", "datetime", "h_p", "ejecta"])
# pd.read_csv("../../13-JMA/eruptlist/JMAexer_190601-210731.csv", index_col=0, header=0).fillna(0)
vent_x = 657324  # m UTM
vent_y = 3495137  # m UTM
vent_z = 1000
K = 100
C = 2.5 * K / 3600 ** 1.5
vel = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.1, 1.3, 1.5,
                1.7, 1.9, 2.2, 2.6, 3, 3.4, 3.8, 4.4, 5.2, 6, 6.8, 7.6, 8.8, 10.4, 12, 13.6, 15.2, 17.6, 20.8])
time_interval = 20  # min
z_interval_plume = 100  # m


def filename_erno(erno):
    direc1 = direc2 + str(erno) + "/traj" + str(erno) + "/"
    ertime = pd.to_datetime(table_det.loc[erno, "ertime"])
    trajlist = glob.glob(direc1 + '*mms-1.csv')
    dirlist = (np.array(
        [file.replace(direc2 + str(erno) + "/traj" + str(erno) + "\\", "")[:5].replace("m", "") for file in
         trajlist]).astype(int)) * 100
    w_ratefilename = direc2 + str(erno) + "/w_rate/w_rate_er" + str(erno) + "_K100_site.csv"
    tpointfilename = direc2 + str(erno) + "/w_rate/tpoint_er" + str(erno) + "_K100_site.csv"
    weightfilename = direc2 + str(erno) + "/w_rate/weight3_er" + str(erno) + "_site.csv"
    return direc1, ertime, dirlist, w_ratefilename, tpointfilename, weightfilename


def output(erno):
    direc1 = filename_erno(erno)[0]
    mh = table_er.loc[erno, "h_p"]
    dirlist = filename_erno(erno)[2]
    # input ejecta and wind and calculate tephra-fall load every v_t interval
    h_seglist = np.arange(vent_z + z_interval_plume, vent_z + mh + z_interval_plume, z_interval_plume).astype(int)
    w_rate = pd.DataFrame()
    tpoint = pd.DataFrame()
    false_sheet = pd.DataFrame(np.zeros((len(vel), len(h_seglist))), index=vel, columns=h_seglist)
    # loop for v_t
    for vt in vel:
        if vt < vel[-1]:
            dirlist_vt = dirlist[
                (dirlist // 1000 == int(vt * 100)) | (dirlist == np.min(dirlist[dirlist // 1000 > vt * 100]))]
        else:
            dirlist_vt = dirlist[dirlist // 1000 == int(vt * 100)]
        traj = pd.read_csv(
            direc1 + str(dirlist_vt[0] // 1000 * 10) + "mms-1.csv", index_col=None)
        for dirN in dirlist_vt[1:]:
            traj = traj.append(
                pd.read_csv(direc1 + str(dirN // 1000 * 10) + "mms-1.csv", index_col=None))
        output_newx = pd.DataFrame(columns=h_seglist, index=np.unique(site["h"]))
        output_newy = pd.DataFrame(columns=h_seglist, index=np.unique(site["h"]))
        output_newt = pd.DataFrame(columns=h_seglist, index=site.index)
        for h in h_seglist:
            choco_flake = traj[(traj["d"] == vt) & (traj["h"] == h)]
            if choco_flake[(choco_flake["d"] == vt) & (choco_flake["h"] == h)]["z0"].min() > 550:
                false_sheet.loc[vt, h] = 1
                output_newx[h] = 0
                output_newy[h] = 0
                output_newt[h] = 10000
                continue
            traj_x = interp1d(choco_flake["z0"], choco_flake["x0"], fill_value="extrapolate")
            traj_y = interp1d(choco_flake["z0"], choco_flake["y0"], fill_value="extrapolate")
            traj_t = interp1d(choco_flake["z0"], choco_flake["t0"], fill_value="extrapolate")
            # [2] By calculating the difference between the coordinates at the segregation height and the coordinates at
            # the surface site altitude, the coordinates of the center of tephra dispersion segregating from each height
            # to each site altitude are obtained and tabulated.
            output_newx[h] = traj_x(np.unique(site["h"]))
            output_newy[h] = traj_y(np.unique(site["h"]))
            output_newt[h] = traj_t(site["h"]) - dt.timedelta(hours=filename_erno(erno)[1].hour % 3,
                                                              minutes=filename_erno(erno)[1].minute).seconds

        def w1(i):
            dx = vent_x + site.loc[i, "x"] - output_newx.loc[int(site.loc[i, "h"]), :str(mh + vent_z)]
            dy = vent_y + site.loc[i, "y"] - output_newy.loc[int(site.loc[i, "h"]), :str(mh + vent_z)]
            t = output_newt.loc[i, :str(mh + vent_z)]
            sigma1 = 1.6 * C * t ** 2.5
            sigma2 = 4 * K * t
            w0 = 1 / (np.maximum(sigma1, sigma2) * 3.141592) * np.exp(
                list(-(dx ** 2 + dy ** 2) / np.maximum(sigma1, sigma2))) * 1000  # [t -> kg]
            return w0

        # [3] load concentration every settling velocity interval. Estimating loads of each site and every v_t interval
        # using 2-D gaussian formula, and then make a table
        output_neww = pd.DataFrame(map(w1, site.index))
        output_newt = output_newt.iloc[np.arange(len(output_neww))[np.max(output_neww, 1) > 1e-15], :]
        output_neww = output_neww.iloc[np.arange(len(output_neww))[np.max(output_neww, 1) > 1e-15], :]
        output_neww["v_t"] = vt
        output_newt["v_t"] = vt
        w_rate = w_rate.append(output_neww)
        tpoint = tpoint.append(output_newt)
    w_rate = w_rate.reset_index().set_index("v_t")
    tpoint = tpoint.reset_index().set_index("v_t")
    false_sheet.to_csv(direc2 + str(erno) + "/w_rate/er" + str(erno) + "_false_sheet_Tephra4D.csv")
    w_rate.to_csv(filename_erno(erno)[3])
    tpoint.to_csv(filename_erno(erno)[4])


def calash(erno):
    # [4] loop every tephra segregation profile: 91 options
    mh = table_er.loc[erno, "h_p"]
    ejection = table_er.loc[erno, "ejecta"]
    shokichi = pd.DataFrame(columns=np.arange(vent_z + z_interval_plume, vent_z + mh + z_interval_plume, z_interval_plume), index=vel)
    vtd = np.array([1 / len(shokichi) for i in range(len(shokichi))])
    tpoint = pd.read_csv(filename_erno(erno)[4], index_col=0).fillna(0)
    w_rate = pd.read_csv(filename_erno(erno)[3], index_col=0).fillna(0)

    def w3(vt):
        w3 = pd.DataFrame(columns=["w1", "vt[m/s]", "h_seg", "point", "seg", "time[min]"])
        # loop for tephra segregation profile
        for seg in [40, 49, 90]:
            if seg <= 50:
                if seg % 10 < 5:
                    cum = [stats.lognorm.cdf(10 * z_interval_plume / mh * j, 1, loc=seg % 10, scale=seg // 10 - 3) for j in
                           range(int(np.ceil(mh / z_interval_plume) + 1))]
                    den = [cum[j + 1] - cum[j] for j in range(int(np.ceil(mh / z_interval_plume)))]
                    for i in range(len(shokichi.columns)):
                        shokichi.iloc[:, i] = den[i] * ejection * np.array(vtd)
                else:
                    cum = [stats.lognorm.cdf(10 * z_interval_plume / mh * j, 1, loc=10 - seg % 10, scale=seg // 10 - 3) for j in
                           range(int(np.ceil(mh / z_interval_plume) + 1))]
                    den = [cum[-j - 1] - cum[-j - 2] for j in range(int(np.ceil(mh / z_interval_plume)))]
                    for i in range(len(shokichi.columns)):
                        shokichi.iloc[:, i] = den[i] * ejection * np.array(vtd)
            elif seg == 90:
                for i in shokichi.columns:
                    shokichi.loc[:, i] = ejection / len(shokichi.columns) * np.array(vtd)
            else:
                print("seg is not accurate" + str(seg))
            w_rate_vt = w_rate[w_rate.index == vt]
            tpoint_vt = tpoint[tpoint.index == vt]
            res = np.array(w_rate_vt.iloc[:, 1:] * shokichi.loc[vt].values)
            result = pd.DataFrame(res.reshape(-1).T, columns=["w1"])
            result["point"] = np.tile(w_rate_vt.iloc[:, 0], (len(w_rate_vt.columns) - 1, 1)).T.reshape(-1)
            result["h_seg"] = np.tile(w_rate_vt.columns[1:].astype(int),
                                      (1, len(result) // (len(w_rate_vt.columns) - 1))).T.reshape(-1)
            time_min = tpoint_vt.iloc[:, 1:].values.reshape(-1).T[result["w1"] > 10 ** -6]
            result = result[result["w1"] > 10 ** -6].fillna(0)
            result["w1"] = (result["w1"] * 10 ** 8).astype(int) / 10 ** 8
            result["time[min]"] = time_min
            del time_min
            result["time[min]"] = (result["time[min]"] / 6).fillna(0).astype(int) / 10
            result["vt[m/s]"] = vt
            result["seg"] = seg
            result.columns = result.columns.astype(str)
            result.index = result["point"]
            w3 = w3.append(result)
        return w3

    weight = pd.concat(list(map(w3, shokichi.index)))
    weight.to_csv(filename_erno(erno)[5], index=False)


for event in table_det.index:
    if table_er.loc[event, "h_p"] == 0:
        continue
    output(event)
    calash(event)
