import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import seaborn as sns
import matplotlib
import statsmodels.api as sm
import streamlit as st
# import pylab as py
import datetime
import seaborn as sns
import time
from matplotlib import rcParams
# from matplotlib.cm import get_cmap


def get_good_naming(naming_path):
    # naming = pd.read_csv(naming_path, sep=";")
    naming = pd.read_csv(naming_path)
    # naming = naming[['team', 'crew_name_STB', 'crew_name_PORT', 'sim_used','end_timestamp', 'csv_file']]
    # naming = naming.dropna(axis=1)
    # naming = naming.drop(index=10)
    naming.end_timestamp = pd.to_datetime(naming.end_timestamp, yearfirst=True)
    naming["start_timestamp"] = naming.groupby(
        "sim_used").shift(1)["end_timestamp"]
    naming = naming.rename(
        columns={"crew_name_STB": "STBD", "crew_name_PORT": "PORT"})
    return naming


def get_good_naming(naming_path):
    naming = pd.read_csv(naming_path)  # , sep=";")
    naming["STBD"] = naming.helm_STB + naming.trimmer_STB
    naming["PORT"] = naming.helm_PORT + naming.trimmer_PORT
    # naming = naming[
    #    ["STBD", "PORT", "sim_used", "start_timestamp", "end_timestamp" ,"csv_file"]
    # ]
    # naming = naming.dropna(axis=1)
    # naming = naming.drop(index=10)
    naming.end_timestamp = pd.to_datetime(naming.end_timestamp, yearfirst=True)
    naming.start_timestamp = pd.to_datetime(
        naming.start_timestamp, yearfirst=True)
    # naming["start_timestamp"] = naming.groupby("sim_used").shift(1)["end_timestamp"]
    # naming = naming.rename(columns={"crew_name_STB": "STBD", "crew_name_PORT": "PORT"})
    return naming


def downsample(df, freq):
    df = df.rolling(30 // freq).mean()[:: 30 // freq]
    return df


def get_translated_names(df, dico):
    dico = dico  # pd.read_csv("name_mapping.csv", sep=";")
    column_mapping = dict(zip(dico["Artermis_name"], dico["perf_name"]))
    df = df.rename(columns=lambda x: x.replace("Boat.", ""))
    df = df.rename(columns=lambda x: x.replace(".", "_"))
    df = df.rename(columns=column_mapping)
    return df


def get_AWA(bsp, twa, tws):
    num = tws * np.sin(twa * np.pi / 180)
    denum = bsp + tws * np.cos(twa * np.pi / 180)
    return np.arctan(num / denum) * 180 / np.pi


def get_AWS(bsp, twa, tws):
    return np.sqrt(
        (bsp + tws * np.cos(twa * np.pi / 180)) ** 2
        + (tws * np.sin(twa * np.pi / 180)) ** 2
    )


def get_sim_crew(tack, datetime, boat, naming):
    sim = naming[naming.sim_used == boat]
    if len(sim[(datetime <= sim.end_timestamp) & (datetime > sim.start_timestamp)]) > 0:
        return sim[(datetime <= sim.end_timestamp) & (datetime > sim.start_timestamp)][
            f"{tack}"
        ].iloc[0]
    else:
        return np.nan


def get_metadata(tack, datetime, boat, naming, names):
    sim = naming[naming.sim_used == boat]
    if len(sim[(datetime <= sim.end_timestamp) & (datetime > sim.start_timestamp)]) > 0:
        return sim[(datetime <= sim.end_timestamp) & (datetime > sim.start_timestamp)][
            f"{names}"
        ].iloc[0]
    else:
        return np.nan


# return sim[(datetime<sim.end_timestamp) & (datetime>sim.start_timestamp)][f'{tack}']#.iloc[0]


def get_one_boat_logs(path, dico, naming, freq):
    dico = dico
    data = pd.DataFrame()
    naming = naming
    i = 1
    for root, dirs, file in os.walk(path):
        for f in file:
            if f.endswith(".csv"):
                path_to_file = os.path.join(root, f)
                df = pd.read_csv(path_to_file, low_memory=True)
                df = downsample(df, freq)

                df["Race"] = i
                datetime_str = path_to_file[-16:-4]
                datetime_obj = pd.to_datetime(
                    datetime_str, format="%y%m%d%H%M%S")
                df["Datetime"] = datetime_obj
                # df["Time"] = pd.to_timedelta(df["Time"], unit="s")
                df["New_datetime"] = df.reset_index().Datetime - pd.to_timedelta(
                    df[::-1].reset_index().Time, unit="s"
                )
                df["New_datetime"][-1] = pd.to_datetime(df.Datetime[0])
                df["Datetime"] = df["New_datetime"]
                # df["Datetime"] = pd.to_datetime(df["Datetime"])
                df["Boat"] = path_to_file[-32:-28]
                i += 1
                data = pd.concat([data, df])
    data = get_translated_names(data, dico)
    data["tack"] = np.where(data["TWA"] > 0, 1, 0)
    data["TACK"] = np.where(data["tack"] > 0, "STBD", "PORT")
    data["Tgt_AWA"] = data["Tgt_AWA"].abs()
    data["Flap"] = np.where(
        data["TWA"] > 0, data["FoilPort_FlapAngle"], data["FoilStbd_FlapAngle"]
    )
    data["FoilCant"] = np.where(
        data["TWA"] > 0, data["FoilPort_Cant"], data["FoilStbd_Cant"]
    )
    data["Rudder_Angle"] = np.where(
        data["TWA"] > 0, data["Rudder_Angle"], -data["Rudder_Angle"]
    )
    data["MainTraveller"] = np.where(
        data["TWA"] > 0, data["MainTraveller"], -data["MainTraveller"]
    )
    data["MastRotation"] = np.where(
        data["TWA"] > 0, data["MastRotation"], -data["MastRotation"]
    )
    data["FoilCant_eff"] = np.where(
        data["TWA"] > 0,
        data["FoilCant"] + data["Heel"],
        data["FoilCant"] - data["Heel"],
    )
    data["Leeway"] = np.where(data["TWA"] > 0, data["Leeway"], -data["Leeway"])
    data["MainFootCamberAngle"] = np.where(
        data["TWA"] > 0, data["MainFootCamberAngle"], -
        data["MainFootCamberAngle"]
    )
    data["MainMidCamber"] = np.where(
        data["TWA"] > 0, data["MainMidCamber"], -data["MainMidCamber"]
    )
    data["MainTwist"] = np.where(
        data["TWA"] > 0, data["MainTwist"], -data["MainTwist"])
    data["JibMidCamber_pc"] = np.where(
        data["TWA"] > 0, data["JibMidCamber_pc"], -data["JibMidCamber_pc"]
    )
    data["JibTwist"] = np.where(
        data["TWA"] > 0, data["JibTwist"], -data["JibTwist"])
    data["MainSheetnoLoad"] = np.where(
        data["TWA"] > 0, data["MainSheetnoLoad"], -data["MainSheetnoLoad"]
    )
    data["JibTrack"] = np.where(
        data["TWA"] > 0, data["JibTrack"], -data["JibTrack"])
    data["MainMidCamber"] = data["MainMidCamber"] * 100
    data.MainSheet = data.MainSheet / 1000
    data.JibCunninghamLoad_kg = data.JibCunninghamLoad_kg / 1000
    data.MainCunninghamLoad_kg = data.MainCunninghamLoad_kg / 1000
    data.MainTraveller = data.MainTraveller * 180 / np.pi
    data["AWA"] = data.apply(lambda x: get_AWA(
        x.BSP, x.TWA, x.TWS_kts), axis=1)
    data["AWA"] = np.where(data["TWA"] > 0, data["AWA"], -data["AWA"])
    data["AWS"] = data.apply(lambda x: get_AWS(
        x.BSP, x.TWA, x.TWS_kts), axis=1)
    data["VMG%"] = (data["VMG"] / data["Tgt_VMG"] * 100).abs()
    data["BSP%"] = data["BSP"] / data["Tgt_BSP"] * 100
    data["CWA%"] = data["CWA"] / data["Tgt_CWA"] * 100
    data["Crew"] = '' #data.apply(
        #lambda x: get_sim_crew(x.TACK, x.Datetime, x.Boat, naming), axis=1
    #)
    for names in naming.columns:
        if names not in [
            "Unnamed: 0",
            "PORT",
            "STBD",
            "start_timestamp",
            "end_timestamp",
            "sim_used",
        ]:
            data[f"{names}"] = data.apply(
                lambda x: get_metadata(
                    x.TACK, x.Datetime, x.Boat, naming, names),
                axis=1,
            )

    return data


def get_one_boat_logs_bis(all_data, dico, naming, freq):
    dico = dico
    data = pd.DataFrame()
    naming = naming
    i = 1
    for file in all_data:
        st.write(f"Reading file : {file.name}")
        time.sleep(1)
        df = pd.read_csv(file, low_memory=True)
        df = downsample(df, freq)

        df["Race"] = i
        datetime_str = file.name[-16:-4]
        datetime_obj = pd.to_datetime(
            datetime_str, format="%y%m%d%H%M%S")
        df["Datetime"] = datetime_obj
        # df["Time"] = pd.to_timedelta(df["Time"], unit="s")
        df["New_datetime"] = df.reset_index().Datetime - pd.to_timedelta(
            df[::-1].reset_index().Time, unit="s"
        )
        df["New_datetime"][-1] = pd.to_datetime(df.Datetime[0])
        df["Datetime"] = df["New_datetime"]
        # df["Datetime"] = pd.to_datetime(df["Datetime"])
        df["Boat"] = file.name[-32:-28]
        i += 1
        data = pd.concat([data, df])
    st.success(f"Done Reading !")
    time.sleep(1)
    st.write(f"Creating perf metrics... (This is the longest operation)")
    time.sleep(1)
    data = get_translated_names(data, dico)
    data["tack"] = np.where(data["TWA"] > 0, 1, 0)
    data["TACK"] = np.where(data["tack"] > 0, "STBD", "PORT")
    data["Tgt_AWA"] = data["Tgt_AWA"].abs()
    data["Flap"] = np.where(
        data["TWA"] > 0, data["FoilPort_FlapAngle"], data["FoilStbd_FlapAngle"]
    )
    data["FoilCant"] = np.where(
        data["TWA"] > 0, data["FoilPort_Cant"], data["FoilStbd_Cant"]
    )
    data["Rudder_Angle"] = np.where(
        data["TWA"] > 0, data["Rudder_Angle"], -data["Rudder_Angle"]
    )
    data["MainTraveller"] = np.where(
        data["TWA"] > 0, data["MainTraveller"], -data["MainTraveller"]
    )
    data["MastRotation"] = np.where(
        data["TWA"] > 0, data["MastRotation"], -data["MastRotation"]
    )
    data["FoilCant_eff"] = np.where(
        data["TWA"] > 0,
        data["FoilCant"] + data["Heel"],
        data["FoilCant"] - data["Heel"],
    )
    data["Leeway"] = np.where(data["TWA"] > 0, data["Leeway"], -data["Leeway"])
    data["MainFootCamberAngle"] = np.where(
        data["TWA"] > 0, data["MainFootCamberAngle"], -
        data["MainFootCamberAngle"]
    )
    data["MainMidCamber"] = np.where(
        data["TWA"] > 0, data["MainMidCamber"], -data["MainMidCamber"]
    )
    data["MainTwist"] = np.where(
        data["TWA"] > 0, data["MainTwist"], -data["MainTwist"])
    data["JibMidCamber_pc"] = np.where(
        data["TWA"] > 0, data["JibMidCamber_pc"], -data["JibMidCamber_pc"]
    )
    data["JibTwist"] = np.where(
        data["TWA"] > 0, data["JibTwist"], -data["JibTwist"])
    data["MainSheetnoLoad"] = np.where(
        data["TWA"] > 0, data["MainSheetnoLoad"], -data["MainSheetnoLoad"]
    )
    data["JibTrack"] = np.where(
        data["TWA"] > 0, data["JibTrack"], -data["JibTrack"])
    data["MainMidCamber"] = data["MainMidCamber"] * 100
    data.MainSheet = data.MainSheet / 1000
    data.JibCunninghamLoad_kg = data.JibCunninghamLoad_kg / 1000
    data.MainCunninghamLoad_kg = data.MainCunninghamLoad_kg / 1000
    data.MainTraveller = data.MainTraveller * 180 / np.pi
    data["AWA"] = data.apply(lambda x: get_AWA(
        x.BSP, x.TWA, x.TWS_kts), axis=1)
    data["AWA"] = np.where(data["TWA"] > 0, data["AWA"], -data["AWA"])
    data["AWS"] = data.apply(lambda x: get_AWS(
        x.BSP, x.TWA, x.TWS_kts), axis=1)
    data["VMG%"] = (data["VMG"] / data["Tgt_VMG"] * 100).abs()
    data["BSP%"] = data["BSP"] / data["Tgt_BSP"] * 100
    data["CWA%"] = data["CWA"] / data["Tgt_CWA"] * 100
    data["Crew"] = '' #data.apply(
        #lambda x: get_sim_crew(x.TACK, x.Datetime, x.Boat, naming), axis=1
    # )
    for names in naming.columns:
        if names not in [
            "Unnamed: 0",
            "PORT",
            "STBD",
            "start_timestamp",
            "end_timestamp",
            "sim_used",
        ]:
            #data[f"{names}"] = data.apply(
            #    lambda x: get_metadata(
            #        x.TACK, x.Datetime, x.Boat, naming, names),
            #    axis=1,
            #)
    st.success(f"All done with creating the logs !")
    time.sleep(1)
    return data


def get_all_logs_(path_rawdata, dico, naming, freq):
    paths = []
    naming = naming
    big_data = pd.DataFrame()
    for root, dirs, file in os.walk(path_rawdata):
        # print(root)
        paths.append(root)
    paths = paths[1:][::-1]
    i = 1
    for path in paths:
        log = get_one_boat_logs(path, dico, naming, freq)
        big_data = pd.concat([big_data, log])
    big_data.drop(columns=["New_datetime"], inplace=True)
    big_data.drop(columns=["index"], inplace=True)
    return big_data


def get_all_logs_v2(path_rawdata, path_dico, path_naming, freq):
    dico = pd.read_csv(path_dico, sep=";")
    naming = get_good_naming(path_naming)
    paths = []
    naming = naming
    big_data = pd.DataFrame()
    for root, dirs, file in os.walk(path_rawdata):
        # print(root)
        paths.append(root)
    paths = paths[1:][::-1]
    i = 1
    for path in paths:
        log = get_one_boat_logs(path, dico, naming, freq)
        big_data = pd.concat([big_data, log])
    big_data.drop(columns=["New_datetime"], inplace=True)
    big_data.drop(columns=["index"], inplace=True)

    return big_data.set_index("Datetime")
