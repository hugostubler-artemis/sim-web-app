import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
import seaborn as sns
import matplotlib
import statsmodels.api as sm
# import pylab as py
import datetime
import seaborn as sns
from matplotlib import rcParams
# from matplotlib.cm import get_cmap
from arrow import get


def upwind_downwind(phase):
    """split between upwind and downwind phases

    Parameters
    ----------
    phase : dataframe
        phases

    Returns
    -------
    2 dataframes
        first is upwind phases, second is downwind phases
    """
    return phase[phase.TWA.abs() < 58], phase[phase.TWA.abs() > 130]


def get_metadata(tack, datetime, boat, naming, names):
    sim = naming[naming.sim_used == boat]
    if len(sim[(datetime <= sim.end_timestamp) & (datetime > sim.start_timestamp)]) > 0:
        return sim[(datetime <= sim.end_timestamp) & (datetime > sim.start_timestamp)][
            f"{names}"
        ].iloc[0]
    else:
        return np.nan


def get_phases(
    df, period, min_bsp, dev_hdg, dev_bsp, perc_min, perc_max, naming, TWA_ref="TWA"
):
    """
    get all the phases of the logs
    Might change the function to get the filters as parameters

    Parameters
    ----------
    df : dataframe
        logs or chunk of logs
    TWA_ref : str, optional
        TWA, by default 'TWA'

    Returns
    -------
    dataframe
        phases
    """

    # df.index = pd.to_datetime(df.DATETIME)
    df.dropna(axis=1, how="all", inplace=True)
    df = df.dropna()
    # df = df.dropna(subset=["BSP"])
    naming = naming
    data = pd.DataFrame()
    df.index = pd.to_datetime(df.Datetime)
    df.index = df.index.dropna()
    df = df.dropna(subset=["Time"])
    essai_1 = list(naming.columns)
    essai_2 = list(df.columns)
    intersection = list(set(essai_1) & set(essai_2))
    intersection.extend(["Datetime", "Boat", "Crew", "TACK"])
    for var in df.columns:
        if var not in intersection:
            data[f"{var}"] = df[f"{var}"].rolling(window=f"{period}s").mean().round(2)
            data[f"dev_{var}"] = df[var].rolling(window=f"{period}s").std().round(2)
        # gradient delta
        if var in [
            "BSP",
            "HullAltitude",
            "HDG",
            "TWA",
            "MainSheet",
            "Heel",
            "Trim",
            "VMG",
            "MainTraveller",
            "Leeway",
            "FoilCant",
            "JibSheetingAngle",
            "Flap",
            "Rudder_Rake",
            "Sink",
            "Rudder_Angle",
            "JibSheetLoad_kg",
        ]:
            data[f"delta_{var}"] = (
                df[var]
                .diff()
                .abs()
                .fillna(0)
                .rolling(window=f"{period}s")
                .max()
                .round(2)
                .fillna(0)
            )

        if var in [
            "BSP",
            "HullAltitude",
            "HDG",
            "TWA",
            "MainSheet",
            "Heel",
            "Trim",
            "VMG",
            "MainTraveller",
            "Leeway",
            "FoilCant",
            "JibSheetingAngle",
            "Flap",
            "Rudder_Rake",
            "Sink",
            "Rudder_Angle",
            "JibSheetLoad_kg",
        ]:
            data[f"delta_mean_{var}"] = (
                df[var]
                .diff()
                .abs()
                .fillna(0)
                .rolling(window=f"{period}s")
                .mean()
                .round(2)
            )

        if var in [
            "BSP",
            "HullAltitude",
            "HDG",
            "TWA",
            "Heel",
            "Trim",
            "VMG",
            "MainTraveller",
            "Leeway",
            "FoilCant",
            "JibSheetingAngle",
            "Rudder_Rake",
            "Sink",
            "Rudder_Angle",
            "JibSheetLoad_kg",
        ]:
            data[f"count_{var}"] = (
                (df[var].diff().abs() > 0.01)
                .rolling(window=f"{period}s")
                .sum()
                .fillna(0)
            )
        if var == "MainSheet":
            data[f"count_{var}"] = (
                (df[var].diff().abs() > 0.002)
                .rolling(window=f"{period}s")
                .sum()
                .fillna(0)
            )
        if var == "Flap":
            data[f"count_{var}"] = (
                (df[var].diff().abs() > 0.1)
                .rolling(window=f"{period}s")
                .sum()
                .fillna(0)
            )

        # if var == 'ANGLE_CA1_deg':
        #    data[f'count_{var}'] = (df[var].diff().abs()>.1).rolling(15, min_periods=5).sum().fillna(0)

        # aggregate sum of delta angles
        if var in [
            "BSP",
            "HullAltitude",
            "HDG",
            "TWA",
            "MainSheet",
            "Heel",
            "Trim",
            "VMG",
            "MainTraveller",
            "Leeway",
            "FoilCant",
            "JibSheetingAngle",
            "Flap",
            "Rudder_Rake",
            "Sink",
            "Rudder_Angle",
            "JibSheetLoad_kg",
        ]:
            data[f"sum_{var}"] = (
                df[var].diff().abs().rolling(window=f"{period}s").sum().round(2)
            )
    data["Boat"] = df.Boat  # .shift(10)
    data["Crew"] = df.Crew  # .shift(10)

    for names in list(set(essai_1) & set(essai_2)):
        if names not in [
            "Unnamed: 0",
            "PORT",
            "STBD",
            "start_timestamp",
            "end_timestamp",
            "sim_used",
        ]:
            # data[names] = (
            #    df[names]
            #    .rolling(window=f"{period}s")
            #    .apply(lambda x: x.dropna().iat[0] if not x.dropna().empty else None)
            # )
            data[names] = list(df[names])  # .reset_index()[f"{names}"]
    # data["type"] = df.type
    data["TACK"] = np.where(data["TWA"] > 0, "STBD", "PORT")

    # data = data[data["HullAltitude"] > 0.2]
    # data = data[data["dev_HullAltitude"] < 0.5]
    # data = data[data['dev_RH_LEE'] < 30]
    data = data[data["dev_BSP"] < dev_bsp]
    # data = data[data['dev_TWS_BOW_SGP_km_h_1'] < 8]
    data = data[data["BSP"] > min_bsp]
    data = data[data["BSP%"] > perc_min]
    data = data[data["BSP%"] < perc_max]
    data = data[data["BSP"] < 100]
    # .  data = data[data['dev_HEADING_deg'] < 4]
    data = data[data["dev_HDG"] < dev_hdg]

    data["Datetime"] = data.index

    # last_valid = data.DATETIME.iloc[0]
    # liste=[]
    # for time in data.DATETIME:
    #   if (time-last_valid).total_seconds()>10:
    #       liste.append(time)
    #       last_valid = time

    # data = data[data.index.isin(liste)]
    if data.empty:
        return data

    liste = []
    last_valid = data.index[0]
    for id, row in data.iterrows():
        delta = (get(id) - get(last_valid)).total_seconds()
        if delta > period - 2:
            liste.append(id)
            last_valid = row.Datetime

    data = data[data.index.isin(liste)]

    data["Heel"] = np.where(data["TWA"] > 0, data["Heel"], -data["Heel"])
    data["phases_duration"] = period
    data["VMG"] = data["VMG"].abs()
    data["TWA"] = data["TWA"].abs()
    return data


def get_all_phases(
    df, period, min_bsp, dev_hdg, dev_bsp, perc_min, perc_max, naming, TWA_ref="TWA"
):
    all_phases = pd.DataFrame()
    for run in [
        value
        for value in df.csv_file.unique()
        if value != np.nan and not isinstance(value, float)
    ]:
        run_df = df[df.csv_file == run]
        phases = get_phases(
            run_df,
            period,
            min_bsp,
            dev_hdg,
            dev_bsp,
            perc_min,
            perc_max,
            naming,
            TWA_ref="TWA",
        )
        all_phases = pd.concat([all_phases, phases])
    return all_phases


def filtering(df, quantile, value):
    """
    functions that filters a dataframe on one column

    Parameters
    ----------
    df : dataframe
        phases
    quantile : float from 0 to 1
        quantile to filter
    value : string
        column to filter

    Returns
    -------
    dataframe
        filtered dataframe
    """
    data = df[
        (df[f"{value}"] > df[f"{value}"].quantile(quantile))
        & (df[f"{value}"] < df[f"{value}"].quantile(1 - quantile))
    ]
    return data[f"{value}"]


def create_dataframe_filtering(df, quantile, L):
    """
    return a recap table of the phases with quantiles 0.25, 0.5 and 0.75 of the filtered phases

    Parameters
    ----------
    df : dataframe
        phases
    quantile : float from 0 to 1
        quantile to filter

    Returns
    -------
    dataframe
        filtered phases recap table
    """

    race_recap = pd.DataFrame(
        index=[
            "avg TWS",
            "avg BSP",
            "avg VMG",
            "avg TWA",
            # "avg AWA",
            "avg mainsheet",
            "avg traveller",
            "avg jib sheet",
            "avg Flap",
            "avg Cant",
            "avg Cant eff",
            "avg Heel",
            "Heel stability",
            "avg Trim",
            "Trim stability",
            "avg Leeway",
            "avg Rudder lift",
            "Flight height",
            "flight stability",
            "bsp stability",
            "twa stability",
            "Rudder num of adjustment",
            "Rudder mean variation",
            "Rudder max variation",
            "Rudder variation sum",
            "Cant num of adjustment",
            "Cant mean variation",
            "Cant max variation",
            "Cant variation sum",
            "Flap num of adjustment",
            "Flap mean variation",
            "Flap max variation",
            "Flap variation sum",
            "Traveller num of adjustmenta",
            "Traveller mean variation",
            "Traveller max variation",
            "Traveller variation sum",
            "Sheet num of adjustment",
            "Sheet mean variation",
            "Sheet max variation",
            "Sheet variation sum",
            "Jib sheet num of adjustment",
            "Jib sheet mean variation",
            "Jib sheet max variation",
            "Jib sheet variation sum",
            "Jib sheet load num of adjustment",
            "Jib sheet load mean variation",
            "Jib sheet load max variation",
            "Jib sheet load variation sum",
        ]
    )

    for quant in L:
        value = np.round(np.min(np.diff(L)) / 2, 2) - 0.0102
        data = df[
            (df.VMG > df.VMG.quantile(quant - value))
            & (df.VMG < df.VMG.quantile(quant + value))
        ]
        race_recap[f"{quant}"] = [
            round(data["TWS_kts"].mean(), 2),
            round(filtering(data, quantile, "BSP%").mean(), 2),
            round(np.mean(np.abs(filtering(data, quantile, "VMG%").median())), 3),
            round(np.mean(np.abs(filtering(data, quantile, "TWA").median())), 3),
            # round(np.mean(np.abs(filtering(data, quantile, "AWA").median())), 3),
            round(
                np.mean(np.abs(filtering(data, quantile, "MainSheet").median())),
                3,
            ),
            round(
                np.mean(np.abs(filtering(data, quantile, "MainTraveller").median())),
                3,
            ),
            round(
                np.mean(np.abs(filtering(data, quantile, "JibSheetingAngle").median())),
                3,
            ),
            round(
                np.mean(np.abs(filtering(data, quantile, "Flap").median())),
                3,
            ),
            round(
                np.mean(np.abs(filtering(data, quantile, "FoilCant").median())),
                3,
            ),
            round(
                np.mean(np.abs(filtering(data, quantile, "FoilCant_eff").median())),
                3,
            ),
            round(filtering(data, quantile, "Heel").median(), 3),
            round(filtering(data, quantile, "Heel").std(), 3),
            round(filtering(data, quantile, "Trim").median(), 3),
            round(filtering(data, quantile, "Trim").std(), 3),
            round(filtering(data, quantile, "Leeway").median(), 3),
            round(filtering(data, quantile, "Rudder_Rake").mean(), 3),
            round(filtering(data, quantile, "HullAltitude").median(), 3),
            round(filtering(data, quantile, "dev_HullAltitude").median() * 100, 3),
            round(filtering(data, quantile, "dev_BSP").median(), 3),
            round(filtering(data, quantile, "dev_TWA").median(), 3),
            round(filtering(data, quantile, "count_Rudder_Angle").median(), 3),
            round(filtering(data, quantile, "delta_mean_Rudder_Angle").median(), 3),
            round(filtering(data, quantile, "delta_Rudder_Angle").median(), 3),
            round(filtering(data, quantile, "sum_Rudder_Angle").median(), 3),
            round(filtering(data, quantile, "count_FoilCant").median(), 3),
            round(filtering(data, quantile, "delta_mean_FoilCant").median(), 3),
            round(filtering(data, quantile, "delta_FoilCant").median(), 3),
            round(filtering(data, quantile, "sum_FoilCant").median(), 3),
            round(filtering(data, quantile, "count_Flap").median(), 3),
            round(filtering(data, quantile, "delta_mean_Flap").median(), 3),
            round(filtering(data, quantile, "delta_Flap").median(), 3),
            round(filtering(data, quantile, "sum_Flap").median(), 3),
            round(filtering(data, quantile, "count_MainTraveller").median(), 3),
            round(
                filtering(data, quantile, "delta_mean_MainTraveller").median(),
                3,
            ),
            round(filtering(data, quantile, "delta_MainTraveller").median(), 3),
            round(filtering(data, quantile, "sum_MainTraveller").median(), 3),
            round(data["count_MainSheet"].mean(), 3),
            round(data["delta_mean_MainSheet"].mean(), 3),
            round(data["delta_MainSheet"].mean(), 3),
            round(data["sum_MainSheet"].mean(), 3),
            round(filtering(data, quantile, "count_JibSheetingAngle").mean(), 3),
            round(filtering(data, quantile, "delta_JibSheetingAngle").mean(), 3),
            round(filtering(data, quantile, "delta_JibSheetingAngle").mean(), 3),
            round(filtering(data, quantile, "sum_JibSheetingAngle").mean(), 3),
            round(filtering(data, quantile, "count_JibSheetLoad_kg").mean(), 3),
            round(filtering(data, quantile, "delta_JibSheetLoad_kg").mean(), 3),
            round(filtering(data, quantile, "delta_JibSheetLoad_kg").mean(), 3),
            round(filtering(data, quantile, "sum_JibSheetLoad_kg").mean(), 3),
        ]

    race_recap["stats"] = race_recap.index
    first_column = race_recap.pop("stats")
    race_recap.insert(0, "stats", first_column)
    race_recap["Targets"] = np.nan
    race_recap["Targets"].loc["avg BSP"] = data.Tgt_BSP.mean()
    race_recap["Targets"].loc["avg TWA"] = data.Tgt_CWA.mean()
    race_recap["Targets"].loc["avg traveller"] = data.Tgt_MainTraveller.mean()

    race_recap["Targets"].loc["avg mainsheet"] = data.Tgt_MainSheet.mean()
    race_recap["Targets"].loc["avg VMG"] = np.abs(data.Tgt_VMG.mean())

    race_recap["Targets"].loc["avg Trim"] = data.Tgt_Trim.mean()

    race_recap["Targets"].loc["avg Leeway"] = data.Tgt_Leeway.mean()
    del race_recap["stats"]
    return race_recap


def create_dataframe_Boat(df, quantile):
    """
    return a recap table of the phases with quantiles 0.25, 0.5 and 0.75 of the filtered phases

    Parameters
    ----------
    df : dataframe
        phases
    quantile : float from 0 to 1
        quantile to filter

    Returns
    -------
    dataframe
        filtered phases recap table
    """

    race_recap = pd.DataFrame(
        index=[
            "avg TWS",
            "avg BSP%",
            "avg VMG%",
            "avg TWA",
            # "avg AWA",
            "avg mainsheet",
            "avg traveller",
            "avg jib sheet",
            "avg Flap",
            "avg Cant",
            "avg Cant eff",
            "avg Heel",
            "Heel stability",
            "avg Trim",
            "Trim stability",
            "avg Leeway",
            "avg Rudder lift",
            "Flight height",
            "flight stability",
            "bsp stability",
            "twa stability",
            "Rudder num of adjustment",
            "Rudder mean variation",
            "Rudder max variation",
            "Rudder variation sum",
            "Cant num of adjustment",
            "Cant mean variation",
            "Cant max variation",
            "Cant variation sum",
            "Flap num of adjustment",
            "Flap mean variation",
            "Flap max variation",
            "Flap variation sum",
            "Traveller num of adjustmenta",
            "Traveller mean variation",
            "Traveller max variation",
            "Traveller variation sum",
            "Sheet num of adjustment",
            "Sheet mean variation",
            "Sheet max variation",
            "Sheet variation sum",
            "Jib sheet num of adjustment",
            "Jib sheet mean variation",
            "Jib sheet max variation",
            "Jib sheet variation sum",
        ]
    )

    for crew in df.Crew:

        data = df[df.Crew == crew]
        race_recap[f"{crew}"] = [
            round(data["TWS_kts"].mean(), 2),
            round(filtering(data, quantile, "BSP%").mean(), 2),
            round(np.mean(np.abs(filtering(data, quantile, "VMG%").median())), 3),
            round(np.mean(np.abs(filtering(data, quantile, "TWA").median())), 3),
            # round(np.mean(np.abs(filtering(data, quantile, "AWA").median())), 3),
            round(
                np.mean(np.abs(filtering(data, quantile, "MainSheet").median())),
                3,
            ),
            round(
                np.mean(np.abs(filtering(data, quantile, "MainTraveller").median())),
                3,
            ),
            round(
                np.mean(np.abs(filtering(data, quantile, "JibSheetingAngle").median())),
                3,
            ),
            round(
                np.mean(np.abs(filtering(data, quantile, "Flap").median())),
                3,
            ),
            round(
                np.mean(np.abs(filtering(data, quantile, "FoilCant").median())),
                3,
            ),
            round(
                np.mean(np.abs(filtering(data, quantile, "FoilCant_eff").median())),
                3,
            ),
            round(filtering(data, quantile, "Heel").median(), 3),
            round(filtering(data, quantile, "Heel").std(), 3),
            round(filtering(data, quantile, "Trim").median(), 3),
            round(filtering(data, quantile, "Trim").std(), 3),
            round(filtering(data, quantile, "Leeway").median(), 3),
            round(filtering(data, quantile, "Rudder_Rake").mean(), 3),
            round(filtering(data, quantile, "HullAltitude").median(), 3),
            round(filtering(data, quantile, "dev_HullAltitude").median() * 100, 3),
            round(filtering(data, quantile, "dev_BSP").median(), 3),
            round(filtering(data, quantile, "dev_TWA").median(), 3),
            round(filtering(data, quantile, "count_Rudder_Angle").median(), 3),
            round(filtering(data, quantile, "delta_mean_Rudder_Angle").median(), 3),
            round(filtering(data, quantile, "delta_Rudder_Angle").median(), 3),
            round(filtering(data, quantile, "sum_Rudder_Angle").median(), 3),
            round(filtering(data, quantile, "count_FoilCant").median(), 3),
            round(filtering(data, quantile, "delta_mean_FoilCant").median(), 3),
            round(filtering(data, quantile, "delta_FoilCant").median(), 3),
            round(filtering(data, quantile, "sum_FoilCant").median(), 3),
            round(filtering(data, quantile, "count_Flap").median(), 3),
            round(filtering(data, quantile, "delta_mean_Flap").median(), 3),
            round(filtering(data, quantile, "delta_Flap").median(), 3),
            round(filtering(data, quantile, "sum_Flap").median(), 3),
            round(filtering(data, quantile, "count_MainTraveller").median(), 3),
            round(
                filtering(data, quantile, "delta_mean_MainTraveller").median(),
                3,
            ),
            round(filtering(data, quantile, "delta_MainTraveller").median(), 3),
            round(filtering(data, quantile, "sum_MainTraveller").median(), 3),
            round(data["count_MainSheet"].mean(), 3),
            round(data["delta_mean_MainSheet"].mean(), 3),
            round(data["delta_MainSheet"].mean(), 3),
            round(data["sum_MainSheet"].mean(), 3),
            round(filtering(data, quantile, "count_JibSheetingAngle").mean(), 3),
            round(filtering(data, quantile, "delta_JibSheetingAngle").mean(), 3),
            round(filtering(data, quantile, "delta_JibSheetingAngle").mean(), 3),
            round(filtering(data, quantile, "sum_JibSheetingAngle").mean(), 3),
        ]

    race_recap["stats"] = race_recap.index
    first_column = race_recap.pop("stats")
    race_recap.insert(0, "stats", first_column)
    race_recap["Targets"] = np.nan
    race_recap["Targets"].loc["avg BSP"] = data.Tgt_BSP.mean()
    race_recap["Targets"].loc["avg TWA"] = data.Tgt_CWA.mean()
    race_recap["Targets"].loc["avg traveller"] = data.Tgt_MainTraveller.mean()

    race_recap["Targets"].loc["avg mainsheet"] = data.Tgt_MainSheet.mean()
    race_recap["Targets"].loc["avg VMG"] = np.abs(data.Tgt_VMG.mean())

    race_recap["Targets"].loc["avg Trim"] = data.Tgt_Trim.mean()

    race_recap["Targets"].loc["avg Leeway"] = data.Tgt_Leeway.mean()

    del race_recap["stats"]
    return race_recap


def create_dataframe_filtering_testing(df, quantile):
    """
    return a recap table of the phases with quantiles 0.25, 0.5 and 0.75 of the filtered phases

    Parameters
    ----------
    df : dataframe
        phases
    quantile : float from 0 to 1
        quantile to filter

    Returns
    -------
    dataframe
        filtered phases recap table
    """

    race_recap = pd.DataFrame(
        index=[
            "avg TWS",
            "avg BSP",
            "avg VMG",
            "avg TWA",
            "avg mainsheet",
            "avg traveller",
            "avg jib sheet",
            "avg Flap",
            "avg Cant",
            "avg Cant eff" "avg Heel",
            "Heel stability",
            "avg Trim",
            "Trim stability",
            "avg Leeway",
            "avg Rudder lift",
            "Flight height",
            "flight stability",
            "bsp stability",
            "twa stability",
            "Cant num of adjustment",
            "Cant mean variation",
            "Cant max variation",
            "Cant variation sum",
            "Flap num of adjustment",
            "Flap mean variation",
            "Flap max variation",
            "Flap variation sum",
            "Traveller num of adjustmenta",
            "Traveller mean variation",
            "Traveller max variation",
            "Traveller variation sum",
            "Sheet num of adjustment",
            "Sheet mean variation",
            "Sheet max variation",
            "Sheet variation sum",
            "Jib sheet num of adjustment",
            "Jib sheet mean variation",
            "Jib sheet max variation",
            "Jib sheet variation sum",
        ]
    )
    data = df.copy()
    race_recap[f"performance"] = [
        round(data["TWS_kts"].mean(), 2),
        round(filtering(data, quantile, "BSP%").mean(), 2),
        round(np.mean(np.abs(filtering(data, quantile, "VMG%").median())), 3),
        round(np.mean(np.abs(filtering(data, quantile, "CWA%").median())), 3),
        round(
            np.mean(np.abs(filtering(data, quantile, "MainSheet").median())),
            3,
        ),
        round(
            np.mean(np.abs(filtering(data, quantile, "MainTraveller").median())),
            3,
        ),
        round(
            np.mean(np.abs(filtering(data, quantile, "JibSheetingAngle").median())),
            3,
        ),
        round(np.mean(np.abs(filtering(data, quantile, "Flap").median())), 3),
        round(np.mean(np.abs(filtering(data, quantile, "FoilCant").median())), 3),
        round(filtering(data, quantile, "Heel").median(), 3),
        round(filtering(data, quantile, "Heel").std(), 3),
        round(filtering(data, quantile, "Trim").median(), 3),
        round(filtering(data, quantile, "Trim").std(), 3),
        round(filtering(data, quantile, "Leeway").abs().median(), 3),
        round(filtering(data, quantile, "Rudder_Rake").mean(), 3),
        round(filtering(data, quantile, "HullAltitude").median(), 3),
        round(filtering(data, quantile, "dev_HullAltitude").median() * 100, 3),
        round(filtering(data, quantile, "dev_BSP").median(), 3),
        round(filtering(data, quantile, "dev_TWA").median(), 3),
        round(filtering(data, quantile, "count_FoilCant").median(), 3),
        round(filtering(data, quantile, "delta_mean_FoilCant").median(), 3),
        round(filtering(data, quantile, "delta_FoilCant").median(), 3),
        round(filtering(data, quantile, "sum_FoilCant").median(), 3),
        round(filtering(data, quantile, "count_Flap").median(), 3),
        round(filtering(data, quantile, "delta_mean_Flap").median(), 3),
        round(filtering(data, quantile, "delta_Flap").median(), 3),
        round(filtering(data, quantile, "sum_Flap").median(), 3),
        round(filtering(data, quantile, "count_MainTraveller").median(), 3),
        round(
            filtering(data, quantile, "delta_mean_MainTraveller").median(),
            3,
        ),
        round(filtering(data, quantile, "delta_MainTraveller").median(), 3),
        round(filtering(data, quantile, "sum_MainTraveller").median(), 3),
        round(data["count_MainSheet"].mean(), 3),
        round(data["delta_mean_MainSheet"].mean(), 3),
        round(data["delta_MainSheet"].mean(), 3),
        round(data["sum_MainSheet"].mean(), 3),
        round(filtering(data, quantile, "count_JibSheetingAngle").mean(), 3),
        round(filtering(data, quantile, "delta_JibSheetingAngle").mean(), 3),
        round(filtering(data, quantile, "delta_JibSheetingAngle").mean(), 3),
        round(filtering(data, quantile, "sum_JibSheetingAngle").mean(), 3),
    ]

    race_recap["stats"] = race_recap.index
    first_column = race_recap.pop("stats")
    race_recap.insert(0, "stats", first_column)
    del race_recap["stats"]
    return race_recap


def get_phases_report(phases):
    return phases[
        [
            "DATETIME",
            "BSP",
            "TWA",
            "Leeway",
            "Heel",
            "TWS",
            "VMG",
            "MainTraveller",
            "sum_MainTraveller",
            "MainSheet",
            "sum_MainSheet",
            "Trim",
            "dev_Trim",
            "JibSheetingAngle",
            "sum_JibSheetingAngle",
            "Rudder_Rake",
            "HullAltitude",
            "dev_HullAltitude",
            "FoilCant",
            "sum_FoilCant",
            "Flap",
            "sum_Flap",
        ]
    ].rename(columns={"dev_HullAltitude": "RH_stab", "dev_Trim": "Trim_stab"})


def get_phase_report(df):
    return (
        create_dataframe_Boat(df, 0.1)
        .style.format(precision=1)
        .background_gradient(cmap="YlGnBu", axis=1)
    )
