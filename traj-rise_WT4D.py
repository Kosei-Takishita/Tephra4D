# 2021_TSP/220909_cal-TSP.pyも参照のこと

import pandas as pd
import numpy as np
import glob
import os
import netCDF4 as nc
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy.interpolate import interp1d
from multiprocessing import Pool
from datetime import timedelta as td

diameter = pd.Series(
    [6.41, 4.76, 4.24, 3.92, 3.70, 3.52, 3.37, 3.22, 3.10, 2.98, 2.87, 2.67, 2.48, 2.31, 2.15, 1.99, 1.70, 1.44, 1.19,
     0.97, 0.76, 0.39, 0.06, -0.24, -0.51, -0.76, -1.23, -1.65, -2.03, -2.66, -3.19, -3.63],
    index=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.4, 2.8, 3.2, 3.6, 4, 4.8,
           5.6, 6.4, 7.2, 8, 9.6, 11.2, 12.8, 16, 19.2, 22.4])
direc1 = 'C:/Users/earth/Documents/Tephra4D/'
direc2 = 'C:/Users/earth/OneDrive/Documents/nied/JAXA/2023/'
dir_app = "C:/Users/earth/Documents/wrf_data/220106/intp_dat/"
fn = 'C:/Users/earth/AppData/Local/Microsoft/Windows/Fonts/mplus1p-medium.ttf'
fn2 = 'C:/Users/earth/AppData/Local/Microsoft/Windows/Fonts/Montserrat-Medium.ttf'
fp = FontProperties(fname=fn2, size=10)
fp2 = FontProperties(fname=fn2, size=9)
fp3 = FontProperties(fname=fn, size=10)

g = 9.81
t_0 = 1200  # K Girault et al. 2014, Woodhouse 2013
r_g_0 = 461  # J/kg/K Girault et al. 2014, Glaze and Baloga 1996
r_a = 287  # J/kg/K Girault et al. 2014
n_0 = 0.03  # wt fraction Girault et al. 2014 -> Woodhouse 2013
u_0 = 100  # 1.8 * np.sqrt(n_0 * r_g_0 * t_0)
rho_p = 2600
rho_a0 = 1.205
c_a = 998  # Woods 1988
alpha_e = 0.15
r_0 = 100  # m
c_p_0 = 2000  # Glaze and Baloga 1996
k_s = 0.09
k_w = 0.9
vent_x = 657324  # 295464
vent_y = 3495137  # 3913472
vent_z = 1000  # 2590
slice = 10  # m

table_er = pd.read_csv("../../../13-JMA/eruptlist/JMAexer_180101-221231.csv", encoding="shift-jis",
                       index_col=0, header=0).fillna(0)
table_er["datetime"] = pd.to_datetime(table_er["datetime"])
grdefdat = pd.read_csv("../../inpfile/grdefdat_seis_220906.csv", index_col=0)
grdefdat.index = pd.to_datetime(grdefdat.index)
range_NS = [0, -1]  # [115, 155]
range_EW = [0, -1]  # [130, 195]
time_interval = 20  # min
time_range = 180  # min
time_slice = time_range // time_interval


class wh13_class(object):
    def __init__(self, erno):
        # 初期値 initial condition
        self.x0n, self.y0n, self.z0n, self.t0n = 83, 72, 0, 0
        ertime = table_er.loc[erno, "datetime"]
        self.h_p = 5000  # table_er.loc[erno, "h_p"]  # conf.loc[unit, "h_p"]
        self.h = vent_z
        self.mer = np.max(np.sum(grdefdat.loc[ertime - td(minutes=16):ertime + td(minutes=118),
                                 ["ejecta_str", "w_seis"]], axis=1)) * 1000 / 60
        self.h_intvl = slice  # m
        self.q_0 = self.mer / np.pi
        self.q = self.mer / np.pi
        self.m = u_0 * self.q
        self.e = self.q * c_p_0 * t_0
        self.theta = np.pi / 2
        self.t = t_0

        windstart_utc = ertime - td(hours=ertime.hour % 3 + 9, minutes=ertime.minute)
        self.t0 = (ertime - windstart_utc - td(hours=9)).seconds
        winddat = nc.Dataset(dir_app + "" + windstart_utc.strftime("%Y-%m-%d_%H%M%S_intp.nc"), "r")
        ertime_jst_10min = ertime - td(minutes=ertime.minute % time_interval)  # 10分単位で指定
        num = int((ertime_jst_10min - windstart_utc - td(hours=9)).seconds / (time_interval * 60))
        self.t0n0 = winddat.variables["time"][num] * 60
        self.vx = np.array(winddat.variables["u"])[num:, :, :, :]
        self.vy = np.array(winddat.variables["v"])[num:, :, :, :]
        self.vz = np.array(winddat.variables["w"])[num:, :, :, :]
        self.X = np.array(winddat.variables["x_utm"]) - vent_x
        self.Y = np.array(winddat.variables["y_utm"]) - vent_y
        alt = np.array(winddat.variables["alt"])
        self.Z = np.array([np.array(winddat.variables["topo"]) + alt[i] for i in range(len(alt))])
        self.p = winddat.variables["p"][num:, :, :, :]
        self.t_pot = winddat.variables["t_pot"][num:, :, :, :]
        self.t_a = self.t_pot[self.t0n, self.z0n, self.y0n, self.x0n] * (
                self.p[self.t0n, self.z0n, self.y0n, self.x0n] / 100000) ** 0.2857
        self.rho_0 = 1 / ((1 - n_0) / rho_p + n_0 * r_g_0 * t_0 / self.p[self.t0n, self.z0n, self.y0n, self.x0n])
        self.v = np.sqrt(self.vx[self.t0n, self.z0n, self.y0n, self.x0n] ** 2 + self.vy[
            self.t0n, self.z0n, self.y0n, self.x0n] ** 2)  # 20 * vent_z / 11000
        self.rho_a = self.p[self.t0n, self.z0n, self.y0n, self.x0n] / r_a / self.t_a

    def rk(self, func1, func2, func3, func4, y1, y2, y3, y4, h):
        k11 = h * func1(y1, y2)
        k12 = h * func2(y1, y2, y3)
        k13 = h * func3(y1, y2, y3)
        k14 = h * func4(y1, y2, y3)

        k21 = h * func1(y1 + k11 / 2, y2 + k12 / 2)
        k22 = h * func2(y1 + k11 / 2, y2 + k12 / 2, y3 + k13 / 2)
        k23 = h * func3(y1 + k11 / 2, y2 + k12 / 2, y3 + k13 / 2)
        k24 = h * func4(y1 + k11 / 2, y2 + k12 / 2, y3 + k13 / 2)

        k31 = h * func1(y1 + k21 / 2, y2 + k22 / 2)
        k32 = h * func2(y1 + k21 / 2, y2 + k22 / 2, y3 + k23 / 2)
        k33 = h * func3(y1 + k21 / 2, y2 + k22 / 2, y3 + k23 / 2)
        k34 = h * func4(y1 + k21 / 2, y2 + k22 / 2, y3 + k23 / 2)

        k41 = h * func1(y1 + k31, y2 + k32)
        k42 = h * func2(y1 + k31, y2 + k32, y3 + k33)
        k43 = h * func3(y1 + k31, y2 + k32, y3 + k33)
        k44 = h * func4(y1 + k31, y2 + k32, y3 + k33)

        y1 += (k11 + 2.0 * k21 + 2.0 * k31 + k41) / 6.0
        y2 += (k12 + 2.0 * k22 + 2.0 * k32 + k42) / 6.0
        y3 += (k13 + 2.0 * k23 + 2.0 * k33 + k43) / 6.0
        y4 += (k14 + 2.0 * k24 + 2.0 * k34 + k44) / 6.0

        return y1, y2, y3, y4

    def interp(self, func):
        return func(self.h)

    # Woodhouse 2013 (12)
    def dq_ds(self, q, m):
        return 2 * self.rho_a * self.u_e() * q / np.sqrt(self.rho() * m)

    # Woodhouse 2013 (13)
    def dm_ds(self, q, m, theta):
        return g * (self.rho_a - self.rho()) * q ** 2 / self.rho() / m * np.sin(theta) + 2 * self.rho_a * q / np.sqrt(
            self.rho() * m) * self.u_e() * self.v * np.cos(theta)

    # Woodhouse 2013 (14)
    def d_theta_ds(self, q, m, theta):
        return g * (self.rho_a - self.rho()) * q ** 2 / self.rho() / m ** 2 * np.cos(theta) - \
            2 * self.rho_a * q / m / np.sqrt(self.rho() * m) * self.u_e() * self.v * np.sin(theta)

    # Woodhouse 2013 (15)
    def de_ds(self, q, m, theta):
        dqds = self.dq_ds(q, m)
        ue = self.u_e()
        return (c_a * self.t_a + ue ** 2 / 2) * dqds + \
            m ** 2 / 2 / q ** 2 * dqds - self.rho_a / self.rho() * q * g * np.sin(theta) - \
            2 * self.rho_a * np.sqrt(m / self.rho()) * ue * self.v * np.cos(theta)

    # Woodhouse 2013 (16)
    def u_e(self):
        return k_s * np.abs(self.m / self.q - self.v * np.cos(self.theta)) + k_w * np.abs(self.v * np.sin(self.theta))

    # Woodhouse 2013 (17)
    def rho(self):
        return 1 / ((1 - self.n()) / rho_p + self.n() * self.r_g() * self.t / self.p[
            self.t0n, self.z0n, self.y0n, self.x0n])

    # Woodhouse 2013 (18)
    def n(self):
        return 1 - (1 - n_0) * self.q_0 / self.q

    # Woodhouse 2013 (19)
    def r_g(self):
        return r_a + (r_g_0 - r_a) * n_0 * (1 - self.n()) / self.n() / (1 - n_0)

    # Woodhouse 2013 (20)
    def c_p(self):
        return c_a + (c_p_0 - c_a) * (1 - self.n()) / (1 - n_0)

    # Woodhouse 2013 (21), 実際には実測値を内挿して用いるのでこの式は不使用。テスト用に残す
    # def t_a(self):
    #     if self.h < 11000:
    #         return self.t_a0 - 6.5e-3 * self.h
    #     elif self.h <= 20000:
    #         return self.t_a0 - 6.5e-3 * 11000
    #     else:
    #         return self.t_a0 - 6.5e-3 * 11000 + 1e-3 * (self.h - 11000)

    # Woodhouse 2013 (22), 実際には実測値を内挿して用いるのでこの式は不使用。テスト用に残す
    # def dp_dz(self):
    #     return -g * self.p[self.t0n, self.z0n, self.y0n, self.x0n] / r_a / self.interp(self.t_func)

    def calc(self):
        prop = pd.DataFrame([[vent_z, vent_x, vent_y, self.t0, self.q, self.m, self.theta, self.e, self.v, self.rho_a,
                              self.m / self.q, self.u_e(), self.p[self.t0n, self.z0n, self.y0n, self.x0n], n_0,
                              self.rho(), r_g_0, self.t, self.x0n, self.y0n, self.z0n, self.t0n]],
                            columns=["z", "x", "y", "t", "q", "m", "theta", "e", "v", "rho_a", "U", "U_e", "p", "n",
                                     "rho", "R_g", "T", "x0n", "y0n", "z0n", "t0n"], index=[0])
        s_j = 0
        while (prop.index[-1] < self.h_p * 1.2 + vent_z) & (self.theta > 0):
            # j = i - 1, j1 = i - 0.5, j2 = i
            z_j, x_j, y_j, t_j, q_j, m_j, theta_j, e_j = prop.iloc[-1, :8]
            s_j += slice
            q_j1, m_j1, theta_j1, e_j1 = self.rk(self.dq_ds, self.dm_ds, self.d_theta_ds, self.de_ds,
                                                 q_j, m_j, theta_j, e_j, slice)
            self.v = np.sqrt(self.vx[self.t0n, self.z0n, self.y0n, self.x0n] ** 2 +
                             self.vy[self.t0n, self.z0n, self.y0n, self.x0n] ** 2)
            x_j1 = x_j + slice * np.cos(theta_j1) * self.vx[self.t0n, self.z0n, self.y0n, self.x0n] / self.v
            y_j1 = y_j + slice * np.cos(theta_j1) * self.vy[self.t0n, self.z0n, self.y0n, self.x0n] / self.v
            z_j1 = np.round(z_j + slice * np.sin(theta_j1))
            self.t0 += slice / (self.m / self.q)

            v_j1 = self.v  # wind speed
            self.q, self.m, self.theta, self.e = q_j1, m_j1, theta_j1, e_j1
            self.t = e_j1 / self.c_p() / q_j1
            self.t_a = self.t_pot[self.t0n, self.z0n, self.y0n, self.x0n] * (
                    self.p[self.t0n, self.z0n, self.y0n, self.x0n] / 100000) ** 0.2857
            self.rho_a = self.p[self.t0n, self.z0n, self.y0n, self.x0n] / r_a / self.t_a
            self.h = z_j1
            x0ny0n = np.where(
                (self.X[1:, 1:] >= 0) & (self.Y[1:, 1:] >= 0) & (self.X[:-1, :-1] < 0) & (self.Y[:-1, :-1] < 0))
            self.x0n, self.y0n = x0ny0n[0][0], x0ny0n[1][0]
            self.z0n = np.where(self.Z[:, self.y0n, self.x0n] >= z_j1)[0][0]
            self.t0n = int((self.t0 - self.t0n0) // (time_interval * 60))

            prop = pd.concat([prop, pd.DataFrame([[z_j1, x_j1, y_j1, self.t0, q_j1, m_j1, theta_j1, e_j1, v_j1,
                                                   self.rho_a, self.m / self.q, self.u_e(),
                                                   self.p[self.t0n, self.z0n, self.y0n, self.x0n], self.n(), self.rho(),
                                                   self.r_g(), self.t, self.x0n, self.y0n, self.z0n, self.t0n]],
                                                 index=[s_j], columns=prop.columns)])
        return prop


def caltraj(erno):
    wh13 = wh13_class(erno)
    traj = wh13.calc()
    traj.to_csv("traj_rise/traj_rise_er" + str(erno) + ".csv")
    # return traj


if __name__ == "__main__":
    caltraj(18242)
    # p = Pool(16)
    # p.map(caltraj, dirlist)


