"""
Copyright (C) 2022  GlaxoSmithKline plc - Patrick Schwab;

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import json
import sys
from itertools import chain
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns


method_display_name_map = {
    "ges": "GES",
    "gies": "GIES",
    "random100": "Random (k=100)",
    "random1000": "Random (k=1000)",
    "random10000": "Random (k=10000)",
    "pc": "PC",
    "notears-mlp-sparse": "NOTEARS (MLP,L1)",
    "notears-lin-sparse": "NOTEARS (Linear,L1)",
    "notears-mlp": "NOTEARS (MLP)",
    "notears-lin": "NOTEARS (Linear)",
    "DCDI-DSF": "DCDI-DSF",
    "DCDI-G": "DCDI-G",
    "grnboost": "GRNBoost"
}

evidence_display_map = {
    "stat_test": "Statistical Edge Test",
    "corum": "Protein Complexes",
    "string_network": "Protein-protein\ninteractions (network)",
    "string_physical": "Protein-protein\ninteractions (physical)",
    # "size": "size",
    "chipseq": "CHIP-Seq",
    "run_time": "Runtime",
    "wasserstein": "Wasserstein"
}

title_display_map = {
    "observational_total_k562": "K562 (observational)",
    "observational_total_rpe1": "RPE1 (observational)",
    "interventional_total_k562": "K562 (interventional)",
    "interventional_total_rpe1": "RPE1 (interventional)",
}

sweep_title_display_map = {
    "obs_subset_sweep_k562": "K562 (observational - varying dataset size)",
    "obs_subset_sweep_rpe1": "RPE1 (observational - varying dataset size)",
    "int_subset_sweep_k562": "K562 (interventional - varying dataset size)",
    "int_subset_sweep_rpe1": "RPE1 (interventional - varying dataset size)"
}

sweep_title_display_map_int = {
    "partial_intervention_sweep_k562": "K562 (interventional - varying intervention set)",
    "partial_intervention_sweep_rpe1": "RPE1 (interventional - varying intervention set)"
}


def json2df_sweep(json_obj) -> pd.DataFrame:
    rows = []
    for method_name in json_obj.keys():
        for step in sorted(map(float, json_obj[method_name].keys())):
            prefix = (method_display_name_map[method_name],
                      step,
                      "mean_wasserstein_distance")
            data_points = json_obj[method_name][str(step)]
            for i, data_point in enumerate(data_points):
                rows.append(prefix + (i, data_point))

    df = pd.DataFrame(data=rows, columns=["method", "step", "metric", "index", "value"])
    return df


def json2df(json_obj) -> pd.DataFrame:
    rows = []
    for method_name in json_obj.keys():
        for evidence_category_name in sorted(json_obj[method_name].keys()):
            if evidence_category_name == "size":
                continue
            for metric_name in sorted(json_obj[method_name][evidence_category_name].keys()):
                data_points = json_obj[method_name][evidence_category_name][metric_name]
                prefix = (method_display_name_map[method_name],
                          evidence_display_map[evidence_category_name],
                          metric_name)
                for i, data_point in enumerate(data_points):
                    rows.append(prefix + (i, data_point))
    df = pd.DataFrame(data=rows, columns=["method", "evidence", "metric", "index", "value"])
    return df


def plot_sweeps(df: pd.DataFrame, file_path: str, title_display_map: dict):
    evidence_types = list(title_display_map.values())
    is_half = (len(evidence_types) // 2) == 1
    fig = plt.figure(constrained_layout=True, figsize=(10, 2 if is_half else 4))
    axes = fig.subplots(len(evidence_types) // 2, 2, sharex=True, sharey=False)
    if not is_half:
        axes = list(chain.from_iterable(axes.tolist()))
    palette = sns.color_palette("pastel")
    markers = ["8", "s", "p", "P", "*", "X", "D"]
    marker_sizes = [8, 8, 10, 10, 17, 14, 8]
    for i, (evidence_type, ax) in enumerate(zip(evidence_types, axes)):
        all_handles = []
        methods = sorted(set(df[(df["evidence"] == evidence_type)]["method"]))
        for j, (method, marker, marker_size) in \
                enumerate(zip(methods, markers, marker_sizes)):
            scatter = ax.scatter(
                df[(df["evidence"] == evidence_type) &
                   (df["metric"] == "mean_wasserstein_distance") &
                   (df["method"] == method)]["step"]*100,
                df[(df["evidence"] == evidence_type) &
                   (df["metric"] == "mean_wasserstein_distance") &
                   (df["method"] == method)]["value"],
                color=palette[j],
                s=marker_size,
                marker=marker,
                edgecolors="black",
                linewidths=0.2,
                zorder=1
            )
            subset = df[(df["evidence"] == evidence_type) &
               (df["metric"] == "mean_wasserstein_distance") &
               (df["method"] == method)][["index", "step", "value"]]
            subset = subset.groupby("step").median().reset_index()
            ax.plot(
                subset["step"]*100,
                subset["value"],
                color=palette[j],
                zorder=1, alpha=0.8, linewidth=1.)
            all_handles += [scatter]
        ax.grid(linestyle="--", color="grey", linewidth=0.25)

        ymin, ymax = ax.get_ylim()
        ax.set_ylim([ymin, ymax*1.1])
        if i > 1 or is_half:
            # ax.set_ylim([0.0, 0.21])
            # Add spacing to match the top axis.
            ax.set_yticklabels([f"{float(label.get_text()):.2f}" for label in ax.get_yticklabels()])
        ax.set_xlim([0.0, 101])

        # ax.set_xlim([0, max_value])
        if i == 1 or i == 3:
            ax.legend(all_handles, methods,
                      loc="center right" if i == 1 and not is_half else "lower right",
                      prop={"family": "Open Sans", "size": 5})
        if i % 2 == 0:
            ax.set_ylabel("Mean Wasserstein\nDistance [unitless]")
        else:
            ax.yaxis.label.set_visible(False)
        if i >= 2:
            ax.set_xlabel("Fraction of Dataset [%]")
        else:
            if is_half:
                ax.set_xlabel("Fraction of Intervention Set [%]")
            else:
                ax.xaxis.label.set_visible(False)
        # plt.title(plot_title)
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")

        ax.text(0.5, 0.95, evidence_type,
                horizontalalignment='center',
                verticalalignment='top',
                transform=ax.transAxes,
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8, pad=2, edgecolor="white"))
    # fig.suptitle(title_display_map[base_name])
    # plt.tight_layout()
    plt.savefig(file_path)
    plt.clf()


def plot_performances(df: pd.DataFrame, base_name: str, file_path: str, x_name: str = "recall", y_name: str = "precision"):
    fig = plt.figure(constrained_layout=True, figsize=(6, 5))
    axes = fig.subplots(3, 2, sharex=True, sharey=True)
    axes = list(chain.from_iterable(axes.tolist()))
    evidence_types = list(evidence_display_map.values())
    x_factor = 100 if x_name == "recall" else 1
    palette = sns.color_palette("pastel")
    markers = ["8", "s", "p", "P", "*", "X", "D"]
    marker_sizes = [8, 8, 10, 10, 17, 14, 8]
    for i, (evidence_type, ax) in enumerate(zip(evidence_types, axes)):
        all_handles = []
        methods = sorted(set(df[(df["evidence"] == evidence_type)]["method"]))
        for j, (method, marker, marker_size) in \
                enumerate(zip(methods, markers, marker_sizes)):

            scatter = ax.scatter(
                df[(df["evidence"] == evidence_type) &
                   (df["metric"] == x_name) &
                   (df["method"] == method)]["value"]*x_factor,
                df[(df["evidence"] == evidence_type) &
                   (df["metric"] == y_name) &
                   (df["method"] == method)]["value"]*100,
                color=palette[j],
                s=marker_size,
                marker=marker,
                edgecolors="black",
                linewidths=0.2,
                zorder=1
            )
            all_handles += [scatter]
        ax.grid(linestyle="--", color="grey", linewidth=0.25)
        ax.set_ylim([0.0, 101])
        ax.set_yticks([0, 25, 50, 75, 100])
        if x_name == "true_positives":
            ax.set_xscale('log')
            ax.set_xlim([1, 10**6])
        else:
            ax.set_xlim([0.0, 101])
            ax.set_xticks([0, 25, 50, 75, 100])
        xlim = ax.get_xlim()
        xlim_rng = xlim[1] - xlim[0]
        for j, (method, marker) in enumerate(zip(methods, markers)):
            avg_x = df[(df["evidence"] == evidence_type) &
                       (df["metric"] == x_name) &
                       (df["method"] == method)]["value"].median() * x_factor
            avg_y = df[(df["evidence"] == evidence_type) &
                       (df["metric"] == y_name) &
                       (df["method"] == method)]["value"].median() * 100
            ax.hlines(y=avg_y, xmin=0, xmax=avg_x, color=palette[j], zorder=0, alpha=0.7, linewidth=1., linestyle="--")
            ax.axvline(x=avg_x, ymin=0, ymax=avg_y / 100, color=palette[j], zorder=0, alpha=0.7, linewidth=1., linestyle="--")

        # ax.set_xlim([0, max_value])
        if i == 1:
            ax.legend(all_handles, methods,
                      loc="center right",
                      prop={"family": "Open Sans", "size": 5})
        if i % 2 == 0:
            ax.set_ylabel("Precision [%]")
        else:
            ax.yaxis.label.set_visible(False)
        if i >= 3:
            if x_name == "recall":
                ax.set_xlabel("Recall [%]")
            else:
                ax.set_xlabel("True Positives [count]")
            ax.xaxis.set_tick_params(labelbottom=True)
        else:
            ax.xaxis.label.set_visible(False)
        # plt.title(plot_title)
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")

        ax.text(0.5, 0.95, evidence_type,
                horizontalalignment='center',
                verticalalignment='top',
                transform=ax.transAxes,
                fontsize=12,
                bbox=dict(facecolor='white', alpha=0.8, pad=2, edgecolor="white"))

    fig.suptitle(title_display_map[base_name])
    fig.delaxes(axes[-1])
    # plt.tight_layout()
    plt.savefig(file_path)
    plt.clf()


def setup(font_dir: str):
    from matplotlib import font_manager

    font_dirs = [os.path.join(font_dir, 'OpenSans')]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)

    # set font
    plt.rcParams['font.family'] = 'Open Sans'


def generate_plots_overall(plot_dir: str, data_dir: str, font_dir: str):
    setup(font_dir)
    files = [
        "observational_total_k562.json",
        "observational_total_rpe1.json",
        "interventional_total_k562.json",
        "interventional_total_rpe1.json"
    ]
    for file_name in files:
        with open(os.path.join(data_dir, file_name), "r") as fp:
            json_obj = json.load(fp)

        df = json2df(json_obj)
        base_name = file_name[:-5]
        outfile_path = os.path.join(plot_dir, f"performances_{base_name}.pdf")
        plot_performances(df, base_name, outfile_path)
        outfile_path_counts = os.path.join(plot_dir, f"performances_{base_name}_counts.pdf")
        plot_performances(df, base_name, outfile_path_counts, x_name="true_positives")


def generate_plots_sweeps(plot_dir: str, data_dir: str):
    files = [
        "obs_subset_sweep_k562.json",
        "obs_subset_sweep_rpe1.json",
        "int_subset_sweep_k562.json",
        "int_subset_sweep_rpe1.json",
    ]
    all_dfs = []
    for file_name in files:
        with open(os.path.join(data_dir, file_name), "r") as fp:
            json_obj = json.load(fp)
        df = json2df_sweep(json_obj)
        base_name = file_name[:-5]
        df["evidence"] = sweep_title_display_map[base_name]
        all_dfs.append(df)
    df = pd.concat(all_dfs)
    outfile_path = os.path.join(plot_dir, f"sweeps_all.pdf")
    plot_sweeps(df, outfile_path, sweep_title_display_map)


def generate_plots_int_sweeps(plot_dir: str, data_dir: str):
    files = [
        "partial_intervention_sweep_k562.json",
        "partial_intervention_sweep_rpe1.json",
    ]
    all_dfs = []
    for file_name in files:
        with open(os.path.join(data_dir, file_name), "r") as fp:
            json_obj = json.load(fp)
        df = json2df_sweep(json_obj)
        base_name = file_name[:-5]
        df["evidence"] = sweep_title_display_map_int[base_name]
        all_dfs.append(df)
    df = pd.concat(all_dfs)
    outfile_path = os.path.join(plot_dir, f"sweeps_partial_interventions.pdf")
    plot_sweeps(df, outfile_path, sweep_title_display_map_int)


def generate_plots(plot_dir: str, data_dir: str, font_dir: str):
    setup(font_dir)
    generate_plots_overall(plot_dir, data_dir, font_dir)
    generate_plots_sweeps(plot_dir, data_dir)
    generate_plots_int_sweeps(plot_dir, data_dir)


if __name__ == "__main__":
    generate_plots(sys.argv[1], sys.argv[2], sys.argv[3])
