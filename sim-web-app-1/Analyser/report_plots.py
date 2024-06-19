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


def hue_regplot(data, x, y, hue, palette, **kwargs):

    regplots = []

    levels = pd.Series(data[hue].unique())
    levels = levels.dropna()

    if palette is None:
        default_colors = get_cmap("tab10")
        palette = {k: default_colors(i) for i, k in enumerate(levels)}

    for key in levels:
        regplots.append(
            sns.regplot(
                x=x,
                y=y,
                data=data[data[hue] == key],
                color=palette[key],
                label=key,
                ci=99,
                **kwargs,
            )
        )

    return regplots


def get_x_y_graph(df, feat1, feat2, hue, ax, order, palette):

    return hue_regplot(
        data=df,
        x=f"{feat1}",
        y=f"{feat2}",
        palette=palette,
        order=order,
        hue=hue,
        scatter_kws={"s": 30},
        ax=ax,
    )


def get_x_y_graph_targets(df, feat1, feat2, hue, ax, order, palette, dico):

    target = "Tgt_" + feat2
    degree = 6
    best_vmg = df[df["VMG%"].abs() > df["VMG%"].abs().quantile(0.95)]
    pairs = pd.Series(best_vmg[f"{hue}"].unique())
    pairs = pairs.dropna()
    x_range = (
        df[f"{feat1}"].min() - df[f"{feat1}"].std() / 10,
        df[f"{feat1}"].max() + df[f"{feat1}"].std() / 10,
    )
    y_range = (
        df[f"{feat2}"].min() - df[f"{feat2}"].std() / 10,
        df[f"{feat2}"].max() + df[f"{feat2}"].std() / 10,
    )
    if target in list(dico.perf_name):
        if target == "Tgt_VMG":
            coefficients = np.polyfit(
                df[f"{feat1}"], df[f"{target}"].abs(), degree)
        if target == "Tgt_TWA":
            coefficients = np.polyfit(
                df[f"{feat1}"], df[f"{target}"].abs(), degree)
        else:
            coefficients = np.polyfit(df[f"{feat1}"], df[f"{target}"], degree)
        polynomial = np.poly1d(coefficients)
        get_x_y_graph(df, feat1, feat2, hue, ax, order, palette)
        # Overlay another curve (for example, a quadratic function)
        x_values = np.linspace(df[f"{feat1}"].min(), df[f"{feat1}"].max())
        # Replace with your own function or data
        y_values = polynomial(x_values)

        # Plot the curve
        ax.plot(x_values, y_values, color="black", linewidth=2.5)

        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        for duo in pairs:
            color = palette[duo]
            ax.scatter(
                best_vmg[best_vmg[f"{hue}"] == duo][f"{feat1}"],
                best_vmg[best_vmg[f"{hue}"] == duo][f"{feat2}"],
                color=color,
                s=160,
                label=duo,
            )
        ax.scatter(
            best_vmg[f"{feat1}"],
            best_vmg[f"{feat2}"],
            marker="*",
            color="yellow",
            s=100,
        )

        ax.legend()
        # plt.show()
    else:
        get_x_y_graph(df, feat1, feat2, hue, ax, order, palette)

        ax.set_xlim(x_range)
        ax.set_ylim(y_range)

        for duo in pairs:
            color = palette[duo]
            ax.scatter(
                best_vmg[best_vmg[f"{hue}"] == duo][f"{feat1}"],
                best_vmg[best_vmg[f"{hue}"] == duo][f"{feat2}"],
                color=color,
                s=160,
                label=duo,
            )
        ax.scatter(
            best_vmg[f"{feat1}"],
            best_vmg[f"{feat2}"],
            marker="*",
            color="yellow",
            s=100,
        )
        ax.legend()


def get_x_y_graph_targets_man(df, feat1, feat2, hue, ax, order, palette):

    target = "target_" + feat2
    degree = 6
    best_vmg = df[df["DistanceMG%"].abs(
    ) > df["DistanceMG%"].abs().quantile(0.95)]
    pairs = pd.Series(best_vmg[f"{hue}"].unique())
    pairs = pairs.dropna()
    x_range = (
        df[f"{feat1}"].min() - df[f"{feat1}"].std() / 10,
        df[f"{feat1}"].max() + df[f"{feat1}"].std() / 10,
    )
    y_range = (
        df[f"{feat2}"].min() - df[f"{feat2}"].std() / 10,
        df[f"{feat2}"].max() + df[f"{feat2}"].std() / 10,
    )
    if target in list(df.columns):
        if target == "Tgt_VMG":
            coefficients = np.polyfit(
                df[f"{feat1}"], df[f"{target}"].abs(), degree)
        if target == "Tgt_TWA":
            coefficients = np.polyfit(
                df[f"{feat1}"], df[f"{target}"].abs(), degree)
        else:
            coefficients = np.polyfit(df[f"{feat1}"], df[f"{target}"], degree)
        polynomial = np.poly1d(coefficients)
        get_x_y_graph(df, feat1, feat2, hue, ax, order, palette)
        # Overlay another curve (for example, a quadratic function)
        x_values = np.linspace(df[f"{feat1}"].min(), df[f"{feat1}"].max())
        # Replace with your own function or data
        y_values = polynomial(x_values)

        # Plot the curve
        ax.plot(x_values, y_values, color="black", linewidth=2.5)

        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        for duo in pairs:
            color = palette[duo]
            ax.scatter(
                best_vmg[best_vmg[f"{hue}"] == duo][f"{feat1}"],
                best_vmg[best_vmg[f"{hue}"] == duo][f"{feat2}"],
                color=color,
                s=160,
                label=duo,
            )
        ax.scatter(
            best_vmg[f"{feat1}"],
            best_vmg[f"{feat2}"],
            marker="*",
            color="yellow",
            s=100,
        )

        ax.legend()
        # plt.show()
    else:
        get_x_y_graph(df, feat1, feat2, hue, ax, order, palette)

        ax.set_xlim(x_range)
        ax.set_ylim(y_range)

        for duo in pairs:
            color = palette[duo]
            ax.scatter(
                best_vmg[best_vmg[f"{hue}"] == duo][f"{feat1}"],
                best_vmg[best_vmg[f"{hue}"] == duo][f"{feat2}"],
                color=color,
                s=160,
                label=duo,
            )
        ax.scatter(
            best_vmg[f"{feat1}"],
            best_vmg[f"{feat2}"],
            marker="*",
            color="yellow",
            s=100,
        )
        ax.legend()


def get_subplots(
    df, variables_to_plot_y, variables_to_plot_x, hue, order, color_list, dico, fig_name
):
    dico = dico
    pairs = pd.Series(df[f"{hue}"].unique())
    pairs = pairs.dropna().sort_values()
    n = len(pairs)

    palette = dict(zip(pairs, color_list[: len(pairs)]))
    num_rows = int(np.round(len(variables_to_plot_y) / 2))
    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(20, 20))

    # Plot regplot on each subplot
    for i, ax in enumerate(axes.flat):
        if i < len(variables_to_plot_y):
            get_x_y_graph_targets(
                df,
                variables_to_plot_x,
                variables_to_plot_y[i],
                hue,
                ax,
                order,
                palette,
                dico,
            )
            ax.set_title(f"{variables_to_plot_y[i]}", fontsize=20)

            ax.set_xlabel(f"{variables_to_plot_x}", fontsize=16)
            ax.set_ylabel(f"{variables_to_plot_y[i]}", fontsize=16)

            # Increase the size of the numbers on the axis
            ax.tick_params(axis='both', which='major', labelsize=25)
            ax.grid(True)
        else:
            pass

    # Adjust layout
    plt.tight_layout()
    plt.legend()
    # Show or save the figure
    # plt.show()
    plt.savefig(f"{fig_name}.png")
    plt.close()


def get_subplots_man(
    df, variables_to_plot_y, variables_to_plot_x, hue, order, color_list, dico, fig_name
):
    dico = dico
    pairs = pd.Series(df[f"{hue}"].unique())
    pairs = pairs.dropna().sort_values()
    n = len(pairs)

    palette = dict(zip(pairs, color_list[: len(pairs)]))
    num_rows = int(np.round(len(variables_to_plot_y) / 2))
    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(20, 20))

    # Plot regplot on each subplot
    for i, ax in enumerate(axes.flat):
        if i < len(variables_to_plot_y):
            get_x_y_graph(  # df, feat1, feat2, hue, ax, order, palette
                df,
                variables_to_plot_x,
                variables_to_plot_y[i],
                hue,
                ax,
                order,
                palette,
            )

            ax.set_title(f"{variables_to_plot_y[i]}", fontsize=20)

            ax.set_xlabel(f"{variables_to_plot_x}", fontsize=16)
            ax.set_ylabel(f"{variables_to_plot_y[i]}", fontsize=16)

            # Increase the size of the numbers on the axis
            ax.tick_params(axis='both', which='major', labelsize=25)
            ax.grid(True)
        else:
            pass

    # Adjust layout
    plt.tight_layout()
    plt.legend()
    # Show or save the figure
    # plt.show()
    plt.savefig(f"{fig_name}.png")
    plt.close()


def get_subplots_man_v2(
    df, variables_to_plot_y, variables_to_plot_x, hue, order, color_list, dico, fig_name
):
    dico = dico
    pairs = pd.Series(df[f"{hue}"].unique())
    pairs = pairs.dropna().sort_values()
    n = len(pairs)

    palette = dict(zip(pairs, color_list[: len(pairs)]))
    num_rows = int(np.round(len(variables_to_plot_y) / 2))
    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(20, 20))

    # Plot regplot on each subplot
    for i, ax in enumerate(axes.flat):
        if i < len(variables_to_plot_y):
            get_x_y_graph_targets_man(  # df, feat1, feat2, hue, ax, order, palette
                df,
                variables_to_plot_x,
                variables_to_plot_y[i],
                hue,
                ax,
                order,
                palette,
            )
            ax.set_title(f"{variables_to_plot_y[i]}")
        else:
            pass

    # Adjust layout
    plt.tight_layout()
    plt.legend()
    # Show or save the figure
    # plt.show()
    plt.savefig(f"{fig_name}.png")
    plt.close()


def get_subplots_v2(df, variables_to_plot, hue, order, color_list, dico, fig_name):
    dico = dico
    pairs = pd.Series(df[f"{hue}"].unique())
    pairs = pairs.dropna()
    n = len(pairs)

    palette = dict(zip(pairs, color_list[: len(pairs)]))
    num_rows = int(np.round(len(variables_to_plot) / 2))
    fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(20, 20))

    # Plot regplot on each subplot
    for i, ax in enumerate(axes.flat):
        get_x_y_graph_targets(
            df,
            variables_to_plot[i][1],
            variables_to_plot[i][0],
            hue,
            ax,
            order,
            palette,
            dico,
        )
        ax.set_title(f"{variables_to_plot[i][1]} vs {variables_to_plot[i][0]}")

    # Adjust layout
    plt.tight_layout()
    plt.legend()
    # Show or save the figure
    # plt.show()
    plt.savefig(f"{fig_name}.png")
    plt.close()
