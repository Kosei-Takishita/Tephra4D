import numpy as np
import datetime as dt
import netCDF4 as nc
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
from pyproj import Proj
import glob

# directory to import and save
direc1 = 'D:/traj200/'
dir_app = "D:/wrf_data/210816/"

r_ns_raw = [100, -100]  # [200, -160]
r_ew_raw = [100, -90]  # [150, -150]
alt_intp = np.arange(0, 6200, 200)  # r stands for range
lon_intp = np.arange(130.575, 130.7375, 0.0025)
lat_intp = np.arange(31.5375, 31.6375, 0.0025)
time_interval = 20  # min
time_range = 180  # min
time_slice = time_range // time_interval
intp_method = "cubic"

e2u_zone = int(divmod(lon_intp[0], 6)[0]) + 31
e2u_conv = Proj(proj='utm', zone=e2u_zone, ellps='WGS84')
lon_intp_mesh, lat_intp_mesh = np.meshgrid(lon_intp, lat_intp)
x_utm, y_utm = np.round(e2u_conv(lon_intp_mesh, lat_intp_mesh), 1)
y_utm = np.where(y_utm < 0, y_utm + 10000000, y_utm)


def cal_tv(t_pot, qv, p):
    work = qv / (1 - qv)
    t = t_pot * (1013.15 * 100 / p) ** -0.2857
    tv = t * (1 + 1.6077 * work) / (1 + work)
    return tv


def interp_wrf_3d(windstart_utc):
    raw_nc = nc.Dataset(dir_app + "raw_dat/" + windstart_utc.strftime("%Y-%m-%d_%H%M%S.nc"), "r")
    # calculate which layer is the maximum to be required
    z_asl_raw = np.array(
        (raw_nc.variables["PH"][:time_slice, :-1, r_ns_raw[0]:r_ns_raw[1], r_ew_raw[0]:r_ew_raw[1]] +
         raw_nc.variables["PH"][:time_slice, 1:, r_ns_raw[0]:r_ns_raw[1], r_ew_raw[0]:r_ew_raw[1]] +
         raw_nc.variables["PHB"][:time_slice, :-1, r_ns_raw[0]:r_ns_raw[1], r_ew_raw[0]:r_ew_raw[1]] +
         raw_nc.variables["PHB"][:time_slice, 1:, r_ns_raw[0]:r_ns_raw[1],
         r_ew_raw[0]:r_ew_raw[1]]) / 9.81 / 2)
    max_iz = np.arange(57)[[np.min(z_asl_raw[:, i, :, :]) < alt_intp[-1] for i in range(57)]][-1] + 1

    # raw data in x, y, z, and t
    u_raw = np.array(
        raw_nc.variables["U"][:time_slice, :max_iz, r_ns_raw[0]:r_ns_raw[1], r_ew_raw[0]:r_ew_raw[1]])
    v_raw = np.array(
        raw_nc.variables["V"][:time_slice, :max_iz, r_ns_raw[0]:r_ns_raw[1], r_ew_raw[0]:r_ew_raw[1]])
    w_raw = np.array(
        raw_nc.variables["W"][:time_slice, :max_iz + 1, r_ns_raw[0]:r_ns_raw[1], r_ew_raw[0]:r_ew_raw[1]])
    lon_raw = np.array(raw_nc.variables["XLONG"][0, r_ns_raw[0]:r_ns_raw[1], r_ew_raw[0]:r_ew_raw[1]])
    lat_raw = np.array(raw_nc.variables["XLAT"][0, r_ns_raw[0]:r_ns_raw[1], r_ew_raw[0]:r_ew_raw[1]])
    t_pot_raw = np.array(raw_nc.variables["T"][:time_slice, :max_iz, r_ns_raw[0]:r_ns_raw[1],
                         r_ew_raw[0]:r_ew_raw[1]]) + 300
    qv_raw = np.array(raw_nc.variables["QVAPOR"][:time_slice, :max_iz, r_ns_raw[0]:r_ns_raw[1],
                      r_ew_raw[0]:r_ew_raw[1]])
    p_raw = (np.array(
        raw_nc.variables["P"][:time_slice, :max_iz, r_ns_raw[0]:r_ns_raw[1],
        r_ew_raw[0]:r_ew_raw[1]]) + np.array(
        raw_nc.variables["PB"][:time_slice, :max_iz, r_ns_raw[0]:r_ns_raw[1], r_ew_raw[0]:r_ew_raw[1]]))

    rho_raw = p_raw / (287.06 * cal_tv(t_pot_raw, qv_raw, p_raw))

    topo_raw = np.array(raw_nc.variables["HGT"][0, r_ns_raw[0]:r_ns_raw[1], r_ew_raw[0]:r_ew_raw[1]])
    z_agl_raw = z_asl_raw[:, :max_iz, :, :] - topo_raw
    yy, xx = np.meshgrid(lat_intp, lon_intp)
    pts = np.array([lon_raw.reshape(-1), lat_raw.reshape(-1)]).T
    topo_intp = np.round(griddata(points=pts, values=topo_raw.reshape(-1), xi=(xx, yy), method=intp_method), 1)

    # interpolate in x and y axis in each time
    for t in range(time_slice):
        print(t, dt.datetime.now())

        def intp_layer_u(iz):
            u_iz = (u_raw[t, iz, :, :-1] + u_raw[t, iz, :, 1:]) / 2
            u_intp_iz_griddata = griddata(points=pts, values=u_iz.reshape(-1), xi=(xx, yy), method=intp_method)
            return u_intp_iz_griddata.T

        def intp_layer_v(iz):
            v_iz = (v_raw[t, iz, :-1, :] + v_raw[t, iz, 1:, :]) / 2
            v_intp_iz_griddata = griddata(points=pts, values=v_iz.reshape(-1), xi=(xx, yy), method=intp_method)
            return v_intp_iz_griddata.T

        def intp_layer_w(iz):
            w_iz = (w_raw[t, iz + 1, :, :] + w_raw[t, iz, :, :]) / 2
            w_intp_iz_griddata = griddata(points=pts, values=w_iz.reshape(-1), xi=(xx, yy), method=intp_method)
            return w_intp_iz_griddata.T

        def intp_layer_z(iz):
            z_agl_iz = z_agl_raw[t, iz, :, :]
            z_intp_iz_griddata = griddata(points=pts, values=z_agl_iz.reshape(-1), xi=(xx, yy), method=intp_method)
            return z_intp_iz_griddata.T

        def intp_layer_rho(iz):
            rho_iz = rho_raw[t, iz, :, :]
            rho_intp_iz_griddata = griddata(points=pts, values=rho_iz.reshape(-1), xi=(xx, yy), method=intp_method)
            return rho_intp_iz_griddata.T

        u_intp_t = np.stack(list(map(intp_layer_u, range(max_iz))))
        v_intp_t = np.stack(list(map(intp_layer_v, range(max_iz))))
        w_intp_t = np.stack(list(map(intp_layer_w, range(max_iz))))
        rho_intp_t = np.stack(list(map(intp_layer_rho, range(max_iz))))
        z_intp_t = np.stack(list(map(intp_layer_z, range(max_iz))))
        r_xx, r_yy = np.meshgrid(range(len(lon_intp)), range(len(lat_intp)))

        # interpolate in z axis
        def intp_grid_dat(iy, ix):
            dat_ixiy = dat[:, iy, ix]
            z_ixiy = z_intp_t[:, iy, ix]
            df_intp_ixiy = interp1d(z_ixiy, dat_ixiy, kind=intp_method, bounds_error=False,
                                    fill_value=(dat_ixiy[0], dat_ixiy[-1]))
            dat_intp_ixiy = df_intp_ixiy(alt_intp)
            return dat_intp_ixiy

        dat = u_intp_t
        u_intp0 = np.stack(map(intp_grid_dat, r_yy.reshape(-1), r_xx.reshape(-1)))
        u_intp1 = u_intp0.T.reshape(1, len(alt_intp), len(lat_intp), len(lon_intp))
        dat = v_intp_t
        v_intp0 = np.stack(map(intp_grid_dat, r_yy.reshape(-1), r_xx.reshape(-1)))
        v_intp1 = v_intp0.T.reshape(1, len(alt_intp), len(lat_intp), len(lon_intp))
        dat = w_intp_t
        w_intp0 = np.stack(map(intp_grid_dat, r_yy.reshape(-1), r_xx.reshape(-1)))
        w_intp1 = w_intp0.T.reshape(1, len(alt_intp), len(lat_intp), len(lon_intp))
        dat = rho_intp_t
        rho_intp0 = np.stack(map(intp_grid_dat, r_yy.reshape(-1), r_xx.reshape(-1)))
        rho_intp1 = rho_intp0.T.reshape(1, len(alt_intp), len(lat_intp), len(lon_intp))

        if t == 0:
            u_intp = u_intp1
            v_intp = v_intp1
            w_intp = w_intp1
            rho_intp = rho_intp1
        else:
            u_intp = np.concatenate([u_intp, u_intp1])
            v_intp = np.concatenate([v_intp, v_intp1])
            w_intp = np.concatenate([w_intp, w_intp1])
            rho_intp = np.concatenate([rho_intp, rho_intp1])

    # export as a netCDF file
    intp_nc = nc.Dataset(dir_app + "intp_dat/" + windstart_utc.strftime("%Y-%m-%d_%H%M%S_intp.nc"), "w",
                         format="NETCDF3_CLASSIC")
    intp_nc.createDimension("time", time_slice)
    intp_nc.createDimension("lon", len(lon_intp))
    intp_nc.createDimension("lat", len(lat_intp))
    intp_nc.createDimension("alt", len(alt_intp))

    intp_nc_time = intp_nc.createVariable("time", np.dtype("int32").char, ("time",))
    intp_nc_time.long_name = "minutes since " + windstart_utc.strftime("%Y/%m/%d %H:%M:%S UTC")
    intp_nc_time.units = "minute"

    intp_nc_lat = intp_nc.createVariable("lat", np.dtype("float32").char, ("lat",))
    intp_nc_lat.long_name = "latitude"
    intp_nc_lat.units = "degree"

    intp_nc_lon = intp_nc.createVariable("lon", np.dtype("float32").char, ("lon",))
    intp_nc_lon.long_name = "longitude"
    intp_nc_lon.units = "degree"

    intp_nc_alt = intp_nc.createVariable("alt", np.dtype("float32").char, ("alt",))
    intp_nc_alt.long_name = "altitude"
    intp_nc_alt.units = "m above ground level"

    intp_nc_x_utm = intp_nc.createVariable("x_utm", np.dtype("float32").char, ("lat", "lon"))
    intp_nc_x_utm.long_name = "x_utm"
    intp_nc_x_utm.units = "m"

    intp_nc_y_utm = intp_nc.createVariable("y_utm", np.dtype("float32").char, ("lat", "lon"))
    intp_nc_y_utm.long_name = "y_utm"
    intp_nc_y_utm.units = "m"

    intp_nc_topo = intp_nc.createVariable("topo", np.dtype("float32").char, ("lat", "lon"))
    intp_nc_topo.long_name = "topography"
    intp_nc_topo.units = "m above sea level"

    intp_nc_u = intp_nc.createVariable("u", np.dtype("float32").char, ("time", "alt", "lat", "lon"))
    intp_nc_u.long_name = "EW wind"
    intp_nc_u.units = "m/s"

    intp_nc_v = intp_nc.createVariable("v", np.dtype("float32").char, ("time", "alt", "lat", "lon"))
    intp_nc_v.long_name = "NS wind"
    intp_nc_v.units = "m/s"

    intp_nc_w = intp_nc.createVariable("w", np.dtype("float32").char, ("time", "alt", "lat", "lon"))
    intp_nc_w.long_name = "UD wind"
    intp_nc_w.units = "m/s"

    intp_nc_rho = intp_nc.createVariable("rho", np.dtype("float32").char, ("time", "alt", "lat", "lon"))
    intp_nc_rho.long_name = "air density"
    intp_nc_rho.units = "kg/m^3"

    intp_nc_time[:] = np.arange(0, time_range, time_interval)
    intp_nc_lat[:] = lat_intp
    intp_nc_lon[:] = lon_intp
    intp_nc_alt[:] = alt_intp
    intp_nc_x_utm[:] = x_utm
    intp_nc_y_utm[:] = y_utm
    intp_nc_topo[:] = topo_intp.T
    intp_nc_u[:] = u_intp
    intp_nc_v[:] = v_intp
    intp_nc_w[:] = w_intp
    intp_nc_rho[:] = rho_intp

    intp_nc.close()


filelist = glob.glob(dir_app + "raw_dat/" + '/*.nc')
for file in filelist:
    windstart_utc = dt.datetime.strptime(file[-20:].replace(".nc", ""), "%Y-%m-%d_%H%M%S")
    interp_wrf_3d(windstart_utc)
