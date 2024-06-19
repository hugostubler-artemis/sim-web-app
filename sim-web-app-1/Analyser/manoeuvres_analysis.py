import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import seaborn as sns
import matplotlib
import statsmodels.api as sm
import pylab as py
import datetime
from arrow import get


def get_man_summary(data, avg_window):
    # avg_window =5
    data.index = data["Datetime"]
    man_summary = pd.DataFrame()
    data["abs_Yaw_rate"] = np.abs(data.HDG.diff() % 350)
    # Find all tack and gybes
    sign = data.TWA > 0
    sign_change = sign != sign.shift(1)
    mans = data[sign_change]
    # filtering the ones where the boat is stopped:
    mans = mans[mans.BSP > 10]
    # if mans.empty:
    #    return mans

    liste = []
    last_valid = mans.index[0]
    for id, row in mans.iterrows():
        delta = (get(id) - get(last_valid)).total_seconds()
        if delta > 12:
            liste.append(id)
            last_valid = row.Datetime

    mans = mans[mans.index.isin(liste)]

    for tman in mans.index.dropna().unique():
        dict = {}
        tack_summary = pd.DataFrame()
        # tman = get(tman)
        tman = get(tman)
        start = tman.shift(seconds=-10).format("YYYY-MM-DD HH:mm:ss")
        stop = tman.shift(seconds=+10).format("YYYY-MM-DD HH:mm:ss")
        tdata = data[(data.index <= stop) & (data.index >= start)]
        entry = tman.shift(seconds=-5).format("YYYY-MM-DD HH:mm:ss")
        exit = tman.shift(seconds=+5).format("YYYY-MM-DD HH:mm:ss")
        build = tman.shift(seconds=+15).format("YYYY-MM-DD HH:mm:ss")
        bigger_build = tman.shift(seconds=+25).format("YYYY-MM-DD HH:mm:ss")

        tdata = data[(data.index <= stop) & (data.index >= start)]
        entry_data = tdata[
            (tdata.index <= entry)
            & (
                tdata.index
                >= tman.shift(seconds=-avg_window - 5).format("YYYY-MM-DD HH:mm:ss")
            )
        ]
        exit_data = tdata[
            (tdata.index >= exit)
            & (
                tdata.index
                <= tman.shift(seconds=+avg_window + 5).format("YYYY-MM-DD HH:mm:ss")
            )
        ]
        tmanoeuvre = tdata[(tdata.index <= exit) & (tdata.index >= entry)]
        bigbuild_data = tdata[(tdata.index >= exit) & (tdata.index <= bigger_build)]
        build_data = tdata[(tdata.index >= exit) & (tdata.index <= build)]

        if len(entry_data) > 0:
            if len(exit_data) > 0:
                if exit_data.BSP.mean() > 10:

                    twa_man = tmanoeuvre.TWA.abs().mean()
                    entry_bsp = round(entry_data.BSP.mean(), 2)
                    # entry_bsp_bd = bd_data.BSP.values
                    # ExitBSP
                    exit_bsp = round(exit_data.BSP.mean(), 2)
                    # Min BSP
                    min_bsp = round(tdata.BSP.min(), 2)
                    # Entry TWA
                    entry_vmg = round(entry_data.VMG.mean(), 1)
                    entry_twa = round(entry_data["TWA"].abs().mean(), 1)
                    # entry_twa_bd = round(bd_data[refTWA].mean(), 2)
                    entry_mainsheet = round(entry_data.MainSheet.abs().mean(), 2)
                    entry_traveller = round(entry_data.MainTraveller.abs().mean(), 2)
                    entry_jib_sheet = round(entry_data.JibSheet.abs().mean(), 2)
                    entry_jib_track = round(entry_data.JibTrack.abs().mean(), 2)
                    entry_rh = round(entry_data.HullAltitude.mean(), 2)

                    entry_pitch = round(entry_data.Trim.mean(), 2)

                    # exit TWA
                    exit_vmg = round(exit_data.VMG.abs().mean(), 1)
                    exit_twa = round(exit_data["TWA"].abs().mean(), 1)
                    exit_mainsheet = round(exit_data.MainSheet.abs().mean(), 2)
                    exit_traveller = round(exit_data.MainTraveller.abs().mean(), 2)
                    exit_jib_sheet = round(exit_data.JibSheet.abs().mean(), 2)
                    exit_jib_track = round(exit_data.JibTrack.abs().mean(), 2)
                    exit_rh = round(exit_data.HullAltitude.mean(), 2)
                    exit_pitch = round(exit_data.Trim.mean(), 2)

                    # Slecting right RH
                    if entry_data.TWA.mean() > 0:
                        rh = "HullAltitude"
                        flap = "FoilStbd_FlapAngle"
                        entry_heel = round(entry_data.Heel.mean(), 2)
                        exit_heel = -round(exit_data.Heel.mean(), 2)
                        turn_heel = -round(tmanoeuvre.Heel.mean(), 2)

                    else:
                        rh = "HullAltitude"
                        flap = "FoilPort_FlapAngle"
                        entry_heel = -round(entry_data.Heel.mean(), 2)
                        exit_heel = round(exit_data.Heel.mean(), 2)
                        turn_heel = round(tmanoeuvre.Heel.mean(), 2)

                        # entry RH
                    if np.sign(entry_data.TWA.mean()) > 0:
                        exit_flap = round(exit_data[flap].abs().mean(), 2)
                        entry_flap = round(entry_data[flap].abs().mean(), 2)
                    else:
                        exit_flap = round(exit_data[flap].abs().mean(), 2)
                        entry_flap = round(entry_data[flap].abs().mean(), 2)

                    min_rh = round(entry_data[rh].min(), 1)

                    max_rh = round(entry_data[rh].max(), 1)
                    rh_delta = round(max_rh - min_rh, 1)

                    # rh_delta_bd = round(max_rh_bd-min_rh_bd,1)

                    # maneuvre minimum RH:
                    turn_min_rh = round(tdata[rh].min(), 1)
                    turn_rh = round(tmanoeuvre[rh].mean(), 2)
                    turn_pitch = round(tmanoeuvre.Trim.mean(), 2)

                    # max yaw_rate

                    # max_yaw_rate = np.round(tdata[tdata.TWA.diff().abs()<100].abs_Yaw_rate.max(), 2)
                    tdata["yaw_rate"] = tdata.TWA.diff().abs()
                    max_yaw_rate = np.round(
                        tdata[tdata.yaw_rate < 100].yaw_rate.rolling(3).mean().max(), 2
                    )
                    dev_yaw_rate = np.round(
                        tdata[
                            tdata.index != tdata.abs_Yaw_rate.idxmax()
                        ].abs_Yaw_rate.std(),
                        2,
                    )

                    # flying tack?
                    min_fly = tdata.HullAltitude.min()
                    if min_fly > 0.2:
                        flying = 1
                    else:
                        flying = 0

                    if len(tdata[tdata.MainFootCamberAngle.diff().abs() > 0]) > 0:
                        timing_pop = get(
                            tdata[tdata.MainFootCamberAngle.diff().abs() > 0].index[0]
                        )

                        poptime = (timing_pop - tman).total_seconds()

                    else:
                        poptime = np.nan

                        # dividing tacks and gybes:
                    if twa_man > 120:
                        man_type = "gybe"
                    elif twa_man < 60:
                        man_type = "tack"
                    else:
                        man_type = np.nan
                        # if abs(entry_twa) > 90:

                        # TWS
                    entry_tws = entry_data.TWS_kts.mean()
                    exit_tws = exit_data.TWS_kts.mean()
                    TWS = round((entry_tws + exit_tws) / 2, 1)

                    # VMG LOSS
                    tdata_vmg = bigbuild_data
                    datetime_series = pd.to_datetime(tdata_vmg.Datetime)
                    time_diffs = datetime_series.diff()

                    # Drop the first entry since it won't have a preceding timestamp to calculate the difference
                    time_diffs = time_diffs.dropna()

                    # Calculate the total number of seconds by summing the timedeltas
                    total_seconds = time_diffs.sum().total_seconds()
                    tdata_vmg["time"] = np.arange(len(tdata_vmg.VMG))
                    # distance_vmg = np.abs(tdata_vmg.VMG.sum()) / 2
                    distance_vmg = (
                        np.abs((tdata_vmg.VMG * time_diffs).sum().total_seconds()) / 2
                    )

                    distance_before = np.abs(entry_data.VMG.mean()) * total_seconds / 2

                    if distance_before - distance_vmg > 0:
                        vmg_loss = distance_before - distance_vmg
                    else:
                        vmg_loss = np.nan

                        # VMG LOSS TARGET

                    distance_target = (
                        np.abs(entry_data.Tgt_VMG.mean()) * total_seconds / 2
                    )

                    if distance_target - distance_vmg > 0:
                        vmg_loss_target = distance_target - distance_vmg
                    else:
                        vmg_loss_target = np.nan

                        # distance indicator

                    first = (tdata.iloc[0].x, tdata.iloc[0].y)
                    last = (tdata.iloc[-1].x, tdata.iloc[-1].y)
                    dist = np.sqrt(
                        (first[0] - last[0]) ** 2 + (first[1] - last[1]) ** 2
                    )

                    build_bsp = build_data.BSP.mean()
                    build_bsp_stab = build_data.BSP.std()
                    build_twa = build_data.TWA.abs().mean()
                    if entry_twa > 0:
                        build_flap_sum = build_data[flap].abs().sum()
                        build_rh = build_data[rh].mean()
                    else:
                        build_flap_sum = build_data[flap].abs().sum()
                        build_rh = build_data[rh].mean()
                    build_traveller_sum = build_data.MainTraveller.diff().abs().sum()
                    build_sheet_sum = build_data.MainSheet.diff().abs().sum()
                    build_jib_sheet_sum = build_data.JibSheet.abs().sum()
                    turn_time = len(tdata[tdata.abs_Yaw_rate > 3])

                    # flight_controller = entry_data.flight_controller.mean()

                    # Tag :

                    # tack velocity_indicator
                    # tack_indicator = vmg_loss * t_90

                    # Entry :
                    if entry_data.TWA.mean() > 0:
                        tack_side = "stbd"
                        tackside = 1
                    else:
                        tack_side = "port"
                        tackside = -1

                    datetime = tdata.Datetime.iloc[0]
                    crew = tdata.Crew.iloc[0]
                    boat = tdata.Boat.iloc[0]

                    dict = {
                        "man_type": man_type,
                        "twa_man": twa_man,
                        "datetime": datetime,
                        "Datetime": datetime,
                        "entry_bsp": entry_bsp,
                        "exit_bsp": exit_bsp,
                        "min_bsp": min_bsp,
                        "entry_vmg": entry_vmg,
                        "entry_twa": entry_twa,
                        "exit_twa": exit_twa,
                        "exit_vmg": exit_vmg,
                        "entry_mainsheet": entry_mainsheet,
                        "entry_traveller": entry_traveller,
                        "entry_flap": entry_flap,
                        "entry_jib_sheet": entry_jib_sheet,
                        "entry_jib_track": entry_jib_track,
                        "entry_rh_stability": rh_delta,
                        "entry_rh": entry_rh,
                        "entry_heel": entry_heel,
                        "entry_pitch": entry_pitch,
                        "exit_traveller": exit_traveller,
                        "exit_mainsheet": exit_mainsheet,
                        "exit_flap": exit_flap,
                        "exit_jib_sheet": exit_jib_sheet,
                        "exit_jib_track": exit_jib_track,
                        "exit_rh": exit_rh,
                        "exit_heel": exit_heel,
                        "exit_pitch": exit_pitch,
                        "max_yaw_rate": max_yaw_rate,
                        # "db_down": db_down,
                        "vmg_loss": vmg_loss,
                        "flying": flying,
                        "tws": TWS,
                        "turn_min_rh": turn_min_rh,
                        # "t_to_lock": t_to_lock,
                        "build_bsp": build_bsp,
                        "build_twa": build_twa,
                        "build_bsp_stab": build_bsp_stab,
                        "build_flap_sum": build_flap_sum,
                        "build_traveller_sum": build_traveller_sum,
                        "build_rh": build_rh,
                        "build_sheet_sum": build_sheet_sum,
                        "build_jib_sheet_sum": build_jib_sheet_sum,
                        "turn_time": turn_time,
                        "dev_yaw_rate": dev_yaw_rate,
                        "turn_rh": turn_rh,
                        "turn_pitch": turn_pitch,
                        "turn_heel": turn_heel,
                        # "exit_awa": exit_awa,
                        "poptime": poptime,
                        # "tack_indicator": tack_indicator,
                        "tack_side": tack_side,
                        "tackside": tackside,
                        "vmg_loss_target": vmg_loss_target,
                        "distance": dist,
                        "crew": crew,
                        "boat": boat,
                        # "t_90":t_90,
                        # "man_id": man_id
                    }
                    if len(dict) > 0:
                        tack_summary = pd.DataFrame(dict, index=[tman])
                        # print(tack_summary)
                        # print(tack_summary)
                        man_summary = pd.concat([man_summary, tack_summary])
                    else:
                        pass
    return man_summary


def get_man_details(
    df: pd.DataFrame,
    timing,
    percent,
    bsp_limit,
    man_list,
    refTWA="TWA",
):
    """
    Returns the table of all the man + the dataframe with time series around

    Parameters
    ----------
    df : pd.DataFrame
        log

    Returns
    -------
    2 pd.DataFrame
        table of the manoeuvres + table of the time series around the manoeuvres
    """

    all_man = pd.DataFrame()  # columns=['type','man_id']

    count = 0

    index_gybe = 0
    index_tack = 0

    man = get_man_summary(df, timing)
    man["Datetime"] = man.index
    man["man_id"] = np.arange(len(man))

    if len(man) > 0:
        for manid in man.man_id.dropna().unique():
            datetime = get(man[man.man_id == manid].Datetime.iloc[0])
            t1 = datetime.shift(seconds=-15).format("YYYY-MM-DD HH:mm:ss")
            t2 = datetime.shift(seconds=+20).format("YYYY-MM-DD HH:mm:ss")
            data = df[(df.Datetime >= t1) & (df.Datetime <= t2)][man_list].reset_index()

            if (
                pd.to_datetime(to_plot.Datetime.max())
                - pd.to_datetime(to_plot.Datetime.min())
            ).total_seconds() > 30:
                if man[man.man_id == manid].man_type.iloc[0] == "gybe":
                    data["man_type"] = man[man.man_id == manid].man_type.iloc[0]
                    data["man_id"] = index_gybe
                    index_gybe += 1
                elif man[man.man_id == manid].man_type.iloc[0] == "tack":
                    data["man_type"] = man[man.man_id == manid].man_type.iloc[0]
                    data["man_id"] = index_tack
                    index_tack += 1
                else:
                    data["man_type"] = np.nan
                    data["man_id"] = np.nan

                if data.Tack.iloc[:4].mean() > 0:
                    data["Old_flap"] = data.FoilPort_FlapAngle
                    data["New_flap"] = data.FoilStbd_FlapAngle
                    # data["Board_drop"] = data.activator_stbd_down
                    # data["Board_up"] = data.activator_port_up
                else:
                    data["Old_flap"] = data.FoilStbd_FlapAngle
                    data["New_flap"] = data.FoilPort_FlapAngle
                    # data["Board_drop"] = data.activator_port_down
                    # data["Board_up"] = data.activator_stbd_up

                data["time"] = np.arange(len(data.x))

                data["man_id"] = manid
                all_man = pd.concat([all_man, data])
                count += 1

    return man, all_man


def transform_tack(TACK):
    if TACK == "PORT":
        return -1
    if TACK == "STBD":
        return 1


def get_man_details_v2(
    df: pd.DataFrame, timing, percent, bsp_limit, man_list, refTWA="TWA"
):
    """
    Returns the table of all the man + the dataframe with time series around

    Parameters
    ----------
    df : pd.DataFrame
        log

    Returns
    -------
    2 pd.DataFrame
        table of the manoeuvres + table of the time series around the manoeuvres
    """

    # columns=['type','man_id']
    df["Tack"] = df.TACK.apply(transform_tack)
    man = pd.DataFrame()
    all_man = pd.DataFrame()
    for boat in df.Boat.unique():
        man = pd.concat([man, get_man_summary(df[df.Boat == boat], timing)])
        # man["datetime"] = man.index
    man["man_id"] = np.arange(len(man))

    for boat in df.Boat.unique():
        filtered_df = df[df.Boat == boat]
        if len(man) > 0:
            for manid in man[man.boat == boat].man_id.dropna().unique():
                datetime = get(man[man.man_id == manid].Datetime.iloc[0])
                t1 = datetime.shift(seconds=-15).format("YYYY-MM-DD HH:mm:ss")
                t2 = datetime.shift(seconds=+20).format("YYYY-MM-DD HH:mm:ss")
                data = filtered_df[
                    (filtered_df.Datetime >= t1) & (filtered_df.Datetime <= t2)
                ][man_list].reset_index()

                if (
                    pd.to_datetime(data.Datetime.max())
                    - pd.to_datetime(data.Datetime.min())
                ).total_seconds() > 30:
                    if man[man.man_id == manid].man_type.iloc[0] == "gybe":
                        data["man_type"] = man[man.man_id == manid].man_type.iloc[0]
                        # data["man_id"] = index_gybe
                        # index_gybe += 1
                    elif man[man.man_id == manid].man_type.iloc[0] == "tack":
                        data["man_type"] = man[man.man_id == manid].man_type.iloc[0]
                        # data["man_id"] = index_tack
                        # index_tack += 1
                    else:
                        data["man_type"] = np.nan
                        data["man_id"] = np.nan

                    if data.Tack.iloc[:4].mean() > 0:
                        data["Old_flap"] = data.FoilPort_FlapAngle
                        data["New_flap"] = data.FoilStbd_FlapAngle
                        # data["Board_drop"] = data.activator_stbd_down
                        # data["Board_up"] = data.activator_port_up
                    else:
                        data["Old_flap"] = data.FoilStbd_FlapAngle
                        data["New_flap"] = data.FoilPort_FlapAngle
                        # data["Board_drop"] = data.activator_port_down
                        # data["Board_up"] = data.activator_stbd_up

                    data["time"] = np.arange(len(data.x))

                    data["man_id"] = manid
                    all_man = pd.concat([all_man, data])

    return man, all_man


def average_plot(all_man, type_of_man, col):
    new_index = all_man[
        (all_man.man_type == f"{type_of_man}")
        & (all_man[f"{col}"] > all_man[f"{col}"].quantile(0.25))
        & (all_man[f"{col}"] < all_man[f"{col}"].quantile(0.75))
    ].man_id.unique()

    for boat in all_man.Crew.unique():
        filtered_dff = all_man[all_man.man_id.isin(new_index)]
        plt.plot(
            filtered_dff[filtered_dff.Crew == boat]
            .groupby("time")
            .mean()
            .rolling(3)
            .mean()[f"{col}"][:-10],
            label=f"{boat}",
        )
        # plt.label=boat
    plt.legend()
    plt.title(f"{col}")
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(6)
    plt.show()
