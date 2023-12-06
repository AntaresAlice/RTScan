#!/usr/bin/env python3

import colorsys
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pandas as pd

from matplotlib.patches import Patch


def scale_color_intensity(color, scale):
    h, l, s = colorsys.rgb_to_hls(*mpl.colors.to_rgb(color))
    return colorsys.hls_to_rgb(h, 1 - scale * (1 - l), s)


DEFAULT_COLORS = [plt.cm.get_cmap('tab10')(i) for i in range(10)]
DEFAULT_SHADES = [1.0 - 0.2 * i for i in range(4)]
DEFAULT_HATCHES = ["", "\\\\\\\\", "...", "xxx"]

# orders a column in a dataframe the same way the passed sequence is ordered
# if the dataframe contains a value that is not in the sequence, it is put at the end
def enforce_value_order(df, col, value_order):
    df = df.copy()
    df["_bar_index"] = df[col].map({value: i for i, value in enumerate(value_order)}).fillna(len(value_order))
    df = df.sort_values(by="_bar_index", kind='mergesort')
    df.drop(["_bar_index"], axis=1, inplace=True)
    return df

# generalized bar plot
# use "same_group" to group bars together
# use "similar_colors" to additionally group by using a similar color
def bar_plot(
        # data
        df, groups, y, same_group=None, similar_colors=None, *,
        y_stack=None, baseline=None, sort_groups=True,
        # figure size
        height=3, width=5,
        # scale options
        log_scale=False, lower_limit=None, upper_limit=None,
        # bar layout
        label_all_bars=False,
        label_clipped_bars=True, clipped_bar_label_size=None,
        missing_bar_label="N/A", missing_bar_label_size="x-small",
        bar_edges=True, group_spacing_multiplier=1.5, color_spacing_multiplier=0,
        bar_colors=DEFAULT_COLORS, bar_shades=DEFAULT_SHADES, bar_hatches=DEFAULT_HATCHES,
        bar_order=None,
        # group labels and axis labels
        group_label_separator=" | ", group_label_prefix="", group_label_postfix="", group_label_text_size=None,
        horizontal_label=None, vertical_label=None, label_rotation=0, label_text_size="large",
        # legend
        legend="auto", legend_columns=1, legend_text_size=None,
        legend_separator=" ", legend_prefix=" ", legend_postfix=" ",
        decomposed_legend=False, decomposed_legend_base_color="gray",
        # stacked bar
        stack_label=None, stack_hatch="////", stack_label_color="white",
        # baseline
        baseline_color="crimson", baseline_label="", baseline_label_text_size=None):

    df = df.copy()

    if groups is not None and sort_groups:
        df = df.sort_values(by=groups, kind='mergesort')

    if same_group is not None and bar_order is not None:
        df = enforce_value_order(df, same_group, bar_order)

    generate_group_labels = True

    # a group can be a combination of multiple attributes or none at all
    if groups is None or groups in ((), []):
        df["group_name_"] = ""
        generate_group_labels = False
    elif isinstance(groups, (list, tuple)):
        df["group_name_"] = group_label_prefix + df[groups[0]].astype(str)
        for i in range(1, len(groups)):
            df["group_name_"] += group_label_separator + df[groups[i]].astype(str)
        df["group_name_"] += group_label_postfix
        if horizontal_label is None:
            horizontal_label = " | ".join(groups)
    else:
        df["group_name_"] = df[groups].astype(str)
        if horizontal_label is None:
            horizontal_label = groups

    unique_groups = list(df["group_name_"].drop_duplicates())

    if legend == "auto":
        legend = same_group is not None or similar_colors is not None or y_stack is not None

    # these are not actually colors, but the unique labels associated with each color
    unique_color_labels = [None] if same_group is None else list(df[same_group].drop_duplicates())
    unique_shade_labels = [None] if similar_colors is None else list(df[similar_colors].drop_duplicates())
    if len(unique_color_labels) > len(bar_colors):
        raise ValueError(f"Cannot provide more than {len(bar_colors)} distinct colors")
    if len(unique_shade_labels) > len(bar_shades):
        raise ValueError(f"Cannot provide more than {len(bar_shades)} distinct shades of the same color")
    if len(unique_shade_labels) > len(bar_hatches):
        raise ValueError(f"Cannot provide more than {len(bar_hatches)} distinct hatches for the same color")
    # these are the actual color values
    bar_colors = [bar_colors[c] for c in unique_color_labels] if isinstance(bar_colors, dict) else bar_colors[:len(unique_color_labels)]
    bar_shades = [bar_shades[s] for s in unique_shade_labels] if isinstance(bar_shades, dict) else bar_shades[:len(unique_shade_labels)]
    bar_hatches = [bar_hatches[s] for s in unique_shade_labels] if isinstance(bar_hatches, dict) else bar_hatches[:len(unique_shade_labels)]

    # hard-coded formatting options
    bar_width = 1
    spacing_between_groups = bar_width * group_spacing_multiplier
    spacing_between_colors = bar_width * color_spacing_multiplier
    color_spacing = bar_width * len(unique_shade_labels) + spacing_between_colors
    group_spacing = color_spacing * len(unique_color_labels) + spacing_between_groups
    label_offset = (len(unique_color_labels) * len(unique_shade_labels) - 1  + (len(unique_color_labels) - 1) * color_spacing_multiplier) * bar_width / 2
    edge_color = "black" if bar_edges else None
    stack_base_label = "base"

    fig, ax = plt.subplots(figsize=(width, height))

    # make any log scale use powers of 10 unless overridden
    if log_scale and len(df) > 0:
        lower_limit = lower_limit if lower_limit is not None else 10 ** math.floor(math.log10(df[y].min()))
        upper_limit = upper_limit if upper_limit is not None else 10 ** math.ceil(math.log10(df[y].max()))

    if lower_limit is not None:
        ax.set_ylim(bottom=lower_limit)
    if upper_limit is not None:
        ax.set_ylim(top=upper_limit)

    # iterate over all unique combinations and check for each combination if only a single element of the dataframe remains
    # this is incredibly naive, but ensures correctness when values are missing or filter conditions are imprecise
    for i, chosen_color in enumerate(unique_color_labels):
        for k, chosen_intensity in enumerate(unique_shade_labels):
            for j, chosen_group in enumerate(unique_groups):

                # filter
                subset = df[df["group_name_"] == chosen_group]
                if same_group is not None:
                    subset = subset[subset[same_group] == chosen_color]
                if similar_colors is not None:
                    subset = subset[subset[similar_colors] == chosen_intensity]

                if len(subset) > 1:
                    distinct_columns = ", ".join(c for c in subset.columns if c.startswith("i_") and len(subset[c].drop_duplicates()) > 1)
                    raise ValueError(f"Identified multiple distinct values in input columns {distinct_columns}. Full data frame:\n{subset.to_string()}")

                ofst = j * group_spacing + i * color_spacing + k * bar_width
                color = scale_color_intensity(bar_colors[i], bar_shades[k])
                hatch = bar_hatches[k]

                # base bar
                val = subset[y].values[0] if len(subset) == 1 else 0
                ax.bar([ofst], [val], width=bar_width, log=log_scale, color=color, edgecolor=edge_color, hatch=hatch)
                top_of_bar = val
                
                # stacked bar
                if y_stack is not None:
                    stacked_val = subset[y_stack].values[0] if len(subset) == 1 else 0
                    ax.bar([ofst], [stacked_val], width=bar_width, bottom=val, log=log_scale, color=color, edgecolor=edge_color, hatch=stack_hatch)
                    top_of_bar += stacked_val

                ymin, ymax = ax.get_ylim()
                # missing value indicator
                if len(subset) == 0:
                    ax.text(ofst, ymin, f"\u2009{missing_bar_label}", horizontalalignment='center', verticalalignment='bottom', size=missing_bar_label_size, rotation=90)

                # clipping label
                if label_all_bars and (upper_limit is None or top_of_bar < upper_limit):
                    ax.text(ofst, top_of_bar, f"\u2009{top_of_bar:.0f}", horizontalalignment='center', verticalalignment='bottom', size=missing_bar_label_size, rotation=90, color="black")

                # clipping label
                if label_clipped_bars and upper_limit is not None and top_of_bar >= upper_limit:
                    ax.text(ofst, ymax, f"{top_of_bar:.0f}\u2009", horizontalalignment='center', verticalalignment='top', size=clipped_bar_label_size, rotation=90, color="white")

    # draw the baseline if provided
    if baseline is not None:
        ax.axhline(y=baseline, linewidth=2, color=baseline_color)
        if baseline_label != "":
            xmin, _ = ax.get_xlim()
            ax.text(xmin, baseline, f"\u2009{baseline_label}", horizontalalignment='left', verticalalignment='bottom', color=baseline_color, size=baseline_label_text_size)

    # legends need to be created for all bars within the same group to distinguish between them
    # in addition, if there are stacked bars, these need labels, too
    if legend:
        legend_patch = [Patch(facecolor=scale_color_intensity(c, s), hatch=h, edgecolor=edge_color) for c in bar_colors for s, h in zip(bar_shades, bar_hatches)]
        if same_group is not None and similar_colors is not None:
            legend_text = [f"{legend_prefix}{c}{legend_separator}{i}{legend_postfix}" for c in unique_color_labels for i in unique_shade_labels]
            # do not combine colors and shades, and instead show them separately
            if decomposed_legend:
                legend_text = unique_color_labels + unique_shade_labels
                legend_patch = [Patch(facecolor=c, edgecolor=edge_color) for c in bar_colors]
                legend_patch += [Patch(facecolor=scale_color_intensity(decomposed_legend_base_color, s), hatch=h, edgecolor=edge_color) for s, h in zip(bar_shades, bar_hatches)]
        elif same_group is not None:
            legend_text = [*unique_color_labels]
        elif similar_colors is not None:
            legend_text = [*unique_shade_labels]
        else:
            legend_text = [stack_base_label]
        if y_stack is not None:
            legend_patch.append(Patch(facecolor=stack_label_color, hatch=stack_hatch, edgecolor=edge_color))
            legend_text.append(stack_label or y_stack)
        ax.legend(legend_patch, legend_text, ncol=legend_columns, fontsize=legend_text_size)

    # only generate one tick per group
    if generate_group_labels:
        ax.set_xticks([i * group_spacing + label_offset for i in range(len(unique_groups))])
        ax.set_xticklabels(unique_groups)
        ax.xaxis.set_tick_params(rotation=label_rotation)
    else:
        ax.set_xticks([])
    ax.tick_params(axis='both', which='major', labelsize=group_label_text_size)

    if horizontal_label is not None and horizontal_label != "":
        ax.set_xlabel(horizontal_label, size=label_text_size)
    if vertical_label is not None and vertical_label != "":
        ax.set_ylabel(vertical_label, size=label_text_size)

    fig.tight_layout(pad=0.05)
    return fig


def prefilter(data, criteria_dict={}, **criteria_kwargs):
    criteria = criteria_dict | criteria_kwargs
    for key, value in criteria.items():
        if isinstance(value, (list, set)):
            data = data[data[key].isin(value)]
        elif callable(value):
            data = data[value(data[key])]
        else:
            data = data[data[key] == value]
    return data


def load_csv_with_inline_labels(file_name):
    data = pd.read_csv(file_name, header=None)
    new_column_names = []
    for column in data.columns:
        label, *_ = data[column][0].split("=")
        data[column] = data[column].str.slice(start=len(label) + 1)
        data[column] = data[column].str.strip()
        try:
            data[column] = pd.to_numeric(data[column])
        except:
            pass
        new_column_names.append(label.strip())
    data.columns = new_column_names
    return data


def load_csv_with_heading(file_name):
    data = pd.read_csv(file_name)
    for column in data.columns:
        try:
            data[column] = data[column].str.strip()
        except:
            pass
    data.columns = [heading.strip() for heading in data.columns]
    return data


def remove_keys(mapping, *keys_to_exclude):
    keys_to_exclude = {*keys_to_exclude}
    return {k: v for k, v in mapping.items() if k not in keys_to_exclude}


LOG_LABEL = "$\mathregular{2^x}$"
NUMBER_OF_INSERTIONS_LABEL = f"Number of insertions [{LOG_LABEL}]"
NUMBER_OF_POINT_QUERIES_LABEL = f"Number of point queries [{LOG_LABEL}]"
THROUGHPUT_LABEL = "Throughput [queries/s]"
QUERY_TIME_LABEL = "Cumulative query time [ms]"
BUILD_TIME_LABEL = "Build time [ms]"
MEMORY_FOOTPRINT_LABEL = "Memory footprint [GB]"


CANONICAL_KEY_MODE_COLORS = {
    "safe": DEFAULT_COLORS[3],
    "unsafe": DEFAULT_COLORS[2],
    "ext": DEFAULT_COLORS[1],
    "3d": DEFAULT_COLORS[0],
}


def plot_design_experiments(file_ext, directory):
    PQ_BUILD_DEFAULT = 26
    RQ_BUILD_DEFAULT = 25
    PROBE_DEFAULT = 27

    os.makedirs(f"{directory}/plots-ex", exist_ok=True)

    # == EX01 ==
    # demonstrate equal probe times for all conversion methods
    # except for extended range, which slows down as the build size increases
    data = load_csv_with_inline_labels(f"{directory}/experiments/int-to-ray.csv")
    criteria = {"i_perpendicular_rays": 1, "i_num_probe_keys_log": PROBE_DEFAULT}
    plot = bar_plot(
        prefilter(data, criteria), "i_num_build_keys_log", "total_probe_time_ms", "i_key_mode",
        #upper_limit=200,
        upper_limit=10000,
        log_scale=True,
        horizontal_label=NUMBER_OF_INSERTIONS_LABEL, vertical_label=QUERY_TIME_LABEL,
        bar_colors=CANONICAL_KEY_MODE_COLORS)
    plot.savefig(f"{directory}/plots-ex/01-int-to-ray.{file_ext}")

    # == EX02 ==
    # demonstrate that exponent shifts do not change the execution time variance for extended mode
    data = load_csv_with_inline_labels(f"{directory}/experiments/exponent-shift.csv")
    criteria = {"i_perpendicular_rays": 1, "i_num_probe_keys_log": PROBE_DEFAULT, "i_exponent_bias": [-20, 0, 20], "i_key_mode": ["ext", "3d"]}
    plot = bar_plot(
        prefilter(data, criteria), "i_num_build_keys_log", "total_probe_time_ms", "i_key_mode", "i_exponent_bias",
        #upper_limit=200,
        upper_limit=10000,
        log_scale=True,
        legend_columns=2, legend_separator=" with bias ", horizontal_label=NUMBER_OF_INSERTIONS_LABEL, vertical_label=QUERY_TIME_LABEL,
        bar_colors=CANONICAL_KEY_MODE_COLORS)
    plot.savefig(f"{directory}/plots-ex/02-exponent-shift.{file_ext}")

    # == EX03 ==
    # demonstrate that key stride influences the execution time variance for extended mode
    data = load_csv_with_inline_labels(f"{directory}/experiments/stride.csv")
    criteria = {"i_perpendicular_rays": 1, "i_num_probe_keys_log": PROBE_DEFAULT, "i_key_stride_log": [0, 1, 2], "i_key_mode": ["ext", "3d"]}
    data["i_key_stride"] = 1 << data["i_key_stride_log"].values
    plot = bar_plot(
        prefilter(data, criteria), "i_num_build_keys_log", "total_probe_time_ms", "i_key_mode", "i_key_stride",
        #upper_limit=200,
        upper_limit=10000,
        log_scale=True,
        legend_columns=2, legend_separator=" with stride ", horizontal_label=NUMBER_OF_INSERTIONS_LABEL, vertical_label=QUERY_TIME_LABEL,
        bar_colors=CANONICAL_KEY_MODE_COLORS)
    plot.savefig(f"{directory}/plots-ex/03-stride.{file_ext}")

    # == EX04 ==
    # demonstrate that parallel/perpendicular makes a difference for all conversion modes
    data = load_csv_with_inline_labels(f"{directory}/experiments/int-to-ray.csv")
    data = data.replace({"i_perpendicular_rays": {0: "parallel from zero", 1: "perpendicular"}})
    criteria = {"i_num_probe_keys_log": PROBE_DEFAULT, "i_num_build_keys_log": lambda x: x <= 24}
    plot = bar_plot(
        prefilter(data, criteria), ["i_num_build_keys_log"], "total_probe_time_ms", "i_key_mode", "i_perpendicular_rays",
        #upper_limit=200,
        upper_limit=1000,
        log_scale=True,
        decomposed_legend=True,
        legend_columns=3, legend_separator=" with ", horizontal_label=NUMBER_OF_INSERTIONS_LABEL, vertical_label=QUERY_TIME_LABEL,
        bar_colors=CANONICAL_KEY_MODE_COLORS)
    plot.savefig(f"{directory}/plots-ex/04-parallel-vs-perpendicular.{file_ext}")

    # == EX05 ==
    # demonstrate that triangles are the fastest primitive and compaction does not impact probe time
    data = load_csv_with_inline_labels(f"{directory}/experiments/primitive.csv")
    data = data.replace({"i_compaction": {0: "uncompacted", 1: "compacted"}})
    criteria = {"i_perpendicular_rays": 1, "i_start_ray_at_zero": 0, "i_probe_mode": "p_sfl", "i_num_probe_keys_log": PROBE_DEFAULT}
    plot = bar_plot(
        prefilter(data, criteria), "i_num_build_keys_log", "total_probe_time_ms", "i_primitive", "i_compaction",
        decomposed_legend=True,
        horizontal_label=NUMBER_OF_INSERTIONS_LABEL, vertical_label=QUERY_TIME_LABEL)
    plot.savefig(f"{directory}/plots-ex/05-primitive-compaction-lookup-time.{file_ext}")

    # == EX06 ==
    # demonstrate that compaction does not significantly influence build time
    data = load_csv_with_inline_labels(f"{directory}/experiments/primitive.csv")
    data["total_build_time_ms"] = data["build_time_ms"] + data["convert_time_ms"]
    criteria = {"i_compaction": 1, "i_perpendicular_rays": 1, "i_start_ray_at_zero": 0, "i_probe_mode": "p_sfl", "i_num_probe_keys_log": PROBE_DEFAULT}
    plot = bar_plot(
        prefilter(data, criteria), "i_num_build_keys_log", "total_build_time_ms", "i_primitive", y_stack="compact_time_ms",
        stack_label="compaction time", horizontal_label=NUMBER_OF_INSERTIONS_LABEL, vertical_label=BUILD_TIME_LABEL)
    plot.savefig(f"{directory}/plots-ex/06-primitive-compaction-build-time.{file_ext}")

    # == EX07 ==
    # demonstrate that triangles take the least amount of space, but only when compacted
    data = load_csv_with_inline_labels(f"{directory}/experiments/primitive.csv")
    data["size_gb"] = data["final_size"] / 1_000_000_000
    data["uncompacted_size_gb"] = data["uncompacted_size"] / 1_000_000_000
    data["size_delta_gb"] = data["uncompacted_size_gb"] - data["size_gb"]
    data = data.replace({"i_compaction": {0: "uncompacted", 1: "compacted"}})
    criteria = {"i_perpendicular_rays": 1, "i_start_ray_at_zero": 0, "i_probe_mode": "p_sfl", "i_num_probe_keys_log": PROBE_DEFAULT}
    plot = bar_plot(
        prefilter(data, criteria), "i_num_build_keys_log", "size_gb", "i_primitive", "i_compaction",
        decomposed_legend=True,
        stack_label="compaction", horizontal_label=NUMBER_OF_INSERTIONS_LABEL, vertical_label="BVH size [GB]")
    plot.savefig(f"{directory}/plots-ex/07-primitive-compaction-size.{file_ext}")

    # == EX08 ==
    # demonstate that more updates negatively influence probe time [rebuild + query as baseline]
    data = load_csv_with_inline_labels(f"{directory}/experiments/updates.csv")
    data["total_build_time_ms"] = data["build_time_ms"] + data["compact_time_ms"] + data["convert_time_ms"]
    data["i_num_updates"] = 1 << data["i_num_updates_log"].values
    criteria = {"i_perpendicular_rays": 1, "i_num_build_keys_log": PQ_BUILD_DEFAULT, "i_num_probe_keys_log": PROBE_DEFAULT, "i_num_updates_log": lambda x: ~x.isin([2, 4])}
    data = prefilter(data, criteria)
    base_build_time, base_probe_time = prefilter(data, {"i_num_updates_log": 0})[["total_build_time_ms", "total_probe_time_ms"]].iloc[0]
    plot = bar_plot(
        data, "i_num_updates", "total_probe_time_ms",
        upper_limit=600,
        baseline=base_build_time + base_probe_time, baseline_label=f"rebuild and query",
        baseline_label_text_size="large",
        clipped_bar_label_size="large",
        horizontal_label=f"Number of updates", vertical_label=QUERY_TIME_LABEL)
    plot.savefig(f"{directory}/plots-ex/08-update-lookup-time.{file_ext}")

    # == EX09 ==
    # demonstate that rqs scale linearly and that starting the ray at zero negatively impacts execution time
    data = load_csv_with_inline_labels(f"{directory}/experiments/range-query-start-ray-at-zero.csv")
    data = data.replace({"i_start_ray_at_zero": {0: "parallel from offset", 1: "parallel from zero"}})
    criteria = {"i_range_query_hit_count_log": lambda x: x >= 0, "i_num_build_keys_log": RQ_BUILD_DEFAULT}
    plot = bar_plot(
        prefilter(data, criteria), "i_range_query_hit_count_log", "total_probe_time_ms", similar_colors="i_start_ray_at_zero",
        log_scale=True,
        horizontal_label=f"Number of qualifying entries [{LOG_LABEL}]", vertical_label=QUERY_TIME_LABEL)
    plot.savefig(f"{directory}/plots-ex/09-ray-origin.{file_ext}")

    # ===================== UNUSED PLOTS =====================

    # demonstrate equal build times for all conversion methods
    data = load_csv_with_inline_labels(f"{directory}/experiments/int-to-ray.csv")
    criteria = {"i_perpendicular_rays": 1, "i_num_probe_keys_log": PROBE_DEFAULT}
    plot = bar_plot(
        prefilter(data, criteria), "i_num_build_keys_log", "build_time_ms", "i_key_mode", y_stack="compact_time_ms",
        stack_label="compaction", horizontal_label=NUMBER_OF_INSERTIONS_LABEL, vertical_label=BUILD_TIME_LABEL)
    plot.savefig(f"{directory}/plots-ex/x-int-to-ray-build-time.{file_ext}")

    # demonstate that updates take the same amount of time
    data = load_csv_with_inline_labels(f"{directory}/experiments/updates.csv")
    data["total_build_time_ms"] = data["build_time_ms"] + data["compact_time_ms"] + data["convert_time_ms"]
    data["total_update_time_ms"] = data["update_time_ms"] + data["update_convert_time_ms"]
    data["i_num_updates"] = 1 << data["i_num_updates_log"].values
    criteria = {"i_perpendicular_rays": 1, "i_num_build_keys_log": PQ_BUILD_DEFAULT, "i_num_probe_keys_log": PROBE_DEFAULT}
    data = prefilter(data, criteria)
    base_build_time = data["total_build_time_ms"].mean()
    plot = bar_plot(
        data, "i_num_updates", "total_update_time_ms",
        baseline=base_build_time, baseline_label="build time",
        horizontal_label="Number of updates", vertical_label="Update time [ms]")
    plot.savefig(f"{directory}/plots-ex/x-update-time.{file_ext}")

    # demonstate that key size does not influence build time
    data = load_csv_with_inline_labels(f"{directory}/experiments/key-size.csv")
    data["total_build_time_ms"] = data["build_time_ms"] + data["convert_time_ms"]
    data = data.replace({"i_large_keys": {0: "32-bit keys", 1: "64-bit keys"}})
    criteria = {"i_num_probe_keys_log": PROBE_DEFAULT}
    plot = bar_plot(
        prefilter(data, criteria), "i_num_build_keys_log", "total_build_time_ms", "i_large_keys", y_stack="compact_time_ms",
        stack_label="compaction", horizontal_label=NUMBER_OF_INSERTIONS_LABEL, vertical_label=BUILD_TIME_LABEL)
    plot.savefig(f"{directory}/plots-ex/x-key-size-build-time.{file_ext}")

    # demonstate that key size does not influence build size
    data = load_csv_with_inline_labels(f"{directory}/experiments/key-size.csv")
    data = data.replace({"i_large_keys": {0: "32-bit keys", 1: "64-bit keys"}})
    data["size_gb"] = data["final_size"] / 1_000_000_000
    data["uncompacted_size_gb"] = data["uncompacted_size"] / 1_000_000_000
    data["size_delta_gb"] = data["uncompacted_size_gb"] - data["size_gb"]
    criteria = {"i_num_probe_keys_log": PROBE_DEFAULT}
    plot = bar_plot(
        prefilter(data, criteria), "i_num_build_keys_log", "size_gb", "i_large_keys", y_stack="size_delta_gb",
        stack_label="compaction", horizontal_label=NUMBER_OF_INSERTIONS_LABEL, vertical_label="BVH size [GB]")
    plot.savefig(f"{directory}/plots-ex/x-key-size-compaction-size.{file_ext}")

    # demonstate that key size does not influence probe time
    data = load_csv_with_inline_labels(f"{directory}/experiments/key-size.csv")
    data = data.replace({"i_large_keys": {0: "32-bit keys", 1: "64-bit keys"}})
    data = data.replace({"i_compaction": {0: "uncompacted", 1: "compacted"}})
    criteria = {"i_num_probe_keys_log": PROBE_DEFAULT}
    plot = bar_plot(
        prefilter(data, criteria), "i_num_build_keys_log", "total_probe_time_ms", "i_large_keys", "i_compaction",
        horizontal_label=NUMBER_OF_INSERTIONS_LABEL, vertical_label=QUERY_TIME_LABEL)
    plot.savefig(f"{directory}/plots-ex/x-key-size-compaction-lookup-time.{file_ext}")

    # demonstrate that perpendicular + ray offset is fastest
    data = load_csv_with_inline_labels(f"{directory}/experiments/primitive.csv")
    data = data.replace({"i_perpendicular_rays": {0: "parallel ray", 1: "perpendicular ray"}})
    data = data.replace({"i_start_ray_at_zero": {0: "from origin", 1: "from zero"}})
    criteria = {"i_num_build_keys_log": PQ_BUILD_DEFAULT, "i_num_probe_keys_log": PROBE_DEFAULT, "i_compaction": 1, "i_probe_mode": "p_sfl"}
    plot = bar_plot(
        prefilter(data, criteria), "i_primitive", "total_probe_time_ms", "i_perpendicular_rays", "i_start_ray_at_zero",
        legend_separator=" ", horizontal_label="", vertical_label=QUERY_TIME_LABEL)
    plot.savefig(f"{directory}/plots-ex/x-parallel-vs-perpendicular-offset.{file_ext}")

    # demonstrate that sorted probes improve execution times
    data = load_csv_with_inline_labels(f"{directory}/experiments/primitive.csv")
    data = data.replace({"i_perpendicular_rays": {0: "par ray", 1: "perp ray"}})
    data = data.replace({"i_probe_mode": {"p_asc": "sort qs", "p_sfl": "shfl qs"}})
    criteria = {"i_num_build_keys_log": PQ_BUILD_DEFAULT, "i_num_probe_keys_log": PROBE_DEFAULT, "i_compaction": 1, "i_start_ray_at_zero": 0}
    plot = bar_plot(
        prefilter(data, criteria), "i_primitive", "total_probe_time_ms", "i_probe_mode", "i_perpendicular_rays",
        legend_separator="/", horizontal_label="", vertical_label=QUERY_TIME_LABEL)
    plot.savefig(f"{directory}/plots-ex/x-parallel-vs-perpendicular-sort.{file_ext}")


POINT_QUERY_DEFAULTS = {"i_log_build_size": 26, "i_log_probe_size": 27, "i_log_num_batches": 0, "i_sort_insert": 0, "i_sort_probe": 0, "i_hit_rate": 1, "i_build_key_uniformity": 1, "i_probe_zipf_coefficient": 0}
RANGE_QUERY_DEFAULTS = {"i_log_build_size": 25, "i_log_probe_size": 27, "i_log_num_batches": 0, "i_sort_insert": 0, "i_sort_probe": 0}

CANONICAL_BASELINE_NAMES = {
    "rx": "RX",
    "rtx_index": "RX",
    "b+": "B+",
    "b_link_tree": "B+",
    "sa": "SA",
    "sorted_array": "SA",
    "wc": "HT",
    "warpcore": "HT",
    "scan": "UA",
}

CANONICAL_BASELINE_COLORS = {
    "RX": DEFAULT_COLORS[2],
    "B+": DEFAULT_COLORS[3],
    "SA": DEFAULT_COLORS[0],
    "HT": DEFAULT_COLORS[1],
    "UA": DEFAULT_COLORS[4],
}

CANONICAL_BASELINE_ORDER = ["HT", "B+", "SA", "UA", "RX"]

def plot_comparison_experiments(file_ext, directory, key_size):
    full_pq = load_csv_with_heading(f"{directory}/point_query_k{key_size}_v32.csv")
    full_rq = load_csv_with_heading(f"{directory}/range_query_k{key_size}_v32.csv")

    os.makedirs(f"{directory}/plots-{key_size}b", exist_ok=True)

    full_pq["i_type"] = full_pq["i_type"].replace(CANONICAL_BASELINE_NAMES)
    full_rq["i_type"] = full_rq["i_type"].replace(CANONICAL_BASELINE_NAMES)
    full_pq = enforce_value_order(full_pq, "i_type", CANONICAL_BASELINE_ORDER)
    full_rq = enforce_value_order(full_rq, "i_type", CANONICAL_BASELINE_ORDER)

    full_pq["gpu_resident_gb"] = full_pq["gpu_resident_bytes"] / 1_000_000_000
    full_pq["build_gb"] = full_pq["build_bytes"] / 1_000_000_000
    full_pq["build_overhead_gb"] = full_pq["build_gb"] - full_pq["gpu_resident_gb"]
    full_pq["throughput"] = (1 << full_pq["i_log_probe_size"].values) / (full_pq["sort_time_ms"] + full_pq["probe_time_ms"])
    full_pq["i_sort_insert_desc"] = full_pq["i_sort_insert"].replace({0: "unsorted inserts", 1: "sorted inserts"})
    full_pq["i_sort_probe_desc"] = full_pq["i_sort_probe"].replace({0: "unsorted queries", 1: "sorted queries"})
    full_pq["i_sort_probe_desc_short"] = full_pq["i_sort_probe"].replace({0: "u", 1: "s"})
    full_pq["i_sort_combined"] = full_pq["i_sort_probe"] * 2 + full_pq["i_sort_insert"]
    full_pq["i_sort_combined_desc"] = full_pq["i_sort_combined"].replace({0: "both unsorted", 1: "sorted inserts", 2: "sorted queries", 3: "both sorted"})
    full_pq["i_num_batches"] = 1 << full_pq["i_log_num_batches"].values
    full_pq["i_key_concentration"] = 1 - full_pq["i_build_key_uniformity"].values

    full_rq["i_log_key_range"] = full_rq["i_log_build_size"].values + full_rq["i_log_key_range_factor"].values
    full_rq["i_range_query_size"] = 1 << full_rq["i_log_range_query_size"].values
    full_rq["i_key_range"] = 1 << full_rq["i_log_key_range"].values
    full_rq["i_log_range_query_hits"] = full_rq["i_log_range_query_size"].values - full_rq["i_log_key_range_factor"].values
    full_rq["probe_time_per_hit"] = full_rq["probe_time_ms"] / (1 << full_rq["i_log_range_query_size"].values)

    # === SIZE ===
    # vary lookup size / measure throughput
    criteria = remove_keys(POINT_QUERY_DEFAULTS, "i_log_probe_size") | {"i_log_probe_size": lambda x: x > 20}
    plot = bar_plot(
        prefilter(full_pq, criteria), "i_log_probe_size", "throughput", "i_type",
        horizontal_label=NUMBER_OF_POINT_QUERIES_LABEL, vertical_label=THROUGHPUT_LABEL,
        bar_colors=CANONICAL_BASELINE_COLORS)
    plot.savefig(f"{directory}/plots-{key_size}b/01-lookup-size-vs-throughput.{file_ext}")

    # === SIZE ===
    # vary build size / measure throughput
    criteria = remove_keys(POINT_QUERY_DEFAULTS, "i_log_build_size")
    plot = bar_plot(
        prefilter(full_pq, criteria), "i_log_build_size", "throughput", "i_type",
        horizontal_label=NUMBER_OF_INSERTIONS_LABEL, vertical_label=THROUGHPUT_LABEL,
        bar_colors=CANONICAL_BASELINE_COLORS)
    plot.savefig(f"{directory}/plots-{key_size}b/02-build-size-vs-throughput.{file_ext}")

    # === SIZE ===
    # vary build size / measure size
    criteria = remove_keys(POINT_QUERY_DEFAULTS, "i_log_build_size")
    plot = bar_plot(
        prefilter(full_pq, criteria), "i_log_build_size", "gpu_resident_gb", "i_type", y_stack="build_overhead_gb",
        stack_label="during build only", horizontal_label=NUMBER_OF_INSERTIONS_LABEL, vertical_label=MEMORY_FOOTPRINT_LABEL,
        bar_colors=CANONICAL_BASELINE_COLORS)
    plot.savefig(f"{directory}/plots-{key_size}b/03-build-size-vs-index-size.{file_ext}")

    # === SIZE ===
    # vary build size / measure build time
    criteria = remove_keys(POINT_QUERY_DEFAULTS, "i_log_build_size")
    plot = bar_plot(
        prefilter(full_pq, criteria), "i_log_build_size", "build_time_ms", "i_type",
        horizontal_label=NUMBER_OF_INSERTIONS_LABEL, vertical_label=BUILD_TIME_LABEL,
        bar_colors=CANONICAL_BASELINE_COLORS)
    plot.savefig(f"{directory}/plots-{key_size}b/04-build-size-vs-build-time.{file_ext}")

    # === ORDERING ===
    # vary build and lookup order / measure lookup time
    criteria = remove_keys(POINT_QUERY_DEFAULTS, "i_sort_probe", "i_sort_insert")
    plot = bar_plot(
        prefilter(full_pq, criteria).sort_values(by="i_sort_combined", kind='mergesort'),
        "i_sort_combined_desc", "probe_time_ms", "i_type", y_stack="sort_time_ms",
        stack_label="sort", horizontal_label="", vertical_label=QUERY_TIME_LABEL, sort_groups=False,
        bar_colors=CANONICAL_BASELINE_COLORS)
    plot.savefig(f"{directory}/plots-{key_size}b/05-combined-order-vs-lookup-time.{file_ext}")

    # === ORDERING ===
    # vary build order / measure build size
    criteria = remove_keys(POINT_QUERY_DEFAULTS, "i_sort_insert")
    plot = bar_plot(
        prefilter(full_pq, criteria), "i_sort_insert_desc", "gpu_resident_gb", "i_type", y_stack="build_overhead_gb",
        stack_label="during build", horizontal_label="", vertical_label=MEMORY_FOOTPRINT_LABEL,
        bar_colors=CANONICAL_BASELINE_COLORS)
    plot.savefig(f"{directory}/plots-{key_size}b/06-sort-build-size.{file_ext}")

    # === ORDERING ===
    # vary build order / measure build time
    criteria = remove_keys(POINT_QUERY_DEFAULTS, "i_sort_insert")
    plot = bar_plot(
        prefilter(full_pq, criteria), "i_sort_insert_desc", "build_time_ms", "i_type",
        horizontal_label="", vertical_label=BUILD_TIME_LABEL,
        bar_colors=CANONICAL_BASELINE_COLORS)
    plot.savefig(f"{directory}/plots-{key_size}b/07-sort-build-time.{file_ext}")

    # === BATCHING ===
    # vary batching / measure lookup time
    criteria = remove_keys(POINT_QUERY_DEFAULTS, "i_log_num_batches", "i_sort_probe")
    plot = bar_plot(
        prefilter(full_pq, criteria), ["i_sort_probe_desc_short", "i_num_batches"], "probe_time_ms", "i_type", y_stack="sort_time_ms",
        width=7,
        group_label_separator=" / ", stack_label="sort", horizontal_label="Query order / Number of batches", vertical_label=QUERY_TIME_LABEL,
        bar_colors=CANONICAL_BASELINE_COLORS)
    plot.savefig(f"{directory}/plots-{key_size}b/08-batches-vs-lookup-time.{file_ext}")

    # === MISSES ===
    # vary hit rate / measure lookup time
    criteria = remove_keys(POINT_QUERY_DEFAULTS, "i_hit_rate", "i_sort_probe")
    plot = bar_plot(
        prefilter(full_pq, criteria), ["i_sort_probe_desc_short", "i_hit_rate"], "probe_time_ms", "i_type", y_stack="sort_time_ms",
        width=7,
        group_label_separator=" / ", stack_label="sort", horizontal_label="Query order / Hit rate", vertical_label=QUERY_TIME_LABEL,
        bar_colors=CANONICAL_BASELINE_COLORS)
    plot.savefig(f"{directory}/plots-{key_size}b/09-hit-rate-vs-lookup-time.{file_ext}")

    # === SELECTIVITY ===
    # vary range query size / measure lookup time
    criteria = RANGE_QUERY_DEFAULTS | {"i_log_key_range_factor": 0}
    plot = bar_plot(
        prefilter(full_rq, criteria), "i_log_range_query_size", "probe_time_ms", "i_type",
        log_scale=True, lower_limit=10, upper_limit=100_000,
        legend=True,
        horizontal_label=f"Number of qualifying entries [{LOG_LABEL}]", vertical_label=QUERY_TIME_LABEL,
        bar_colors=CANONICAL_BASELINE_COLORS)
    plot.savefig(f"{directory}/plots-{key_size}b/10-range-query-size-vs-lookup-time.{file_ext}")

    # === SPARSITY ===
    # vary key range / measure lookup time
    criteria = RANGE_QUERY_DEFAULTS | {"i_log_range_query_size": 10}
    plot = bar_plot(
        prefilter(full_rq, criteria), "i_log_key_range", "probe_time_ms", "i_type",
        log_scale=True, lower_limit=10, upper_limit=100_000,
        horizontal_label=f"Maximum key [{LOG_LABEL}]", vertical_label=QUERY_TIME_LABEL,
        bar_colors=CANONICAL_BASELINE_COLORS)
    plot.savefig(f"{directory}/plots-{key_size}b/11-key-range-vs-lookup-time.{file_ext}")

    # === SPARSITY ===
    # keep hits constant / vary key range / measure lookup time
    criteria = RANGE_QUERY_DEFAULTS | {"i_log_range_query_hits": 2}
    plot = bar_plot(
        prefilter(full_rq, criteria), ["i_log_range_query_size", "i_log_key_range"], "probe_time_ms", "i_type",
        group_label_separator=" / ", horizontal_label=f"Range query span (s) [{LOG_LABEL}] / Maximum key [{LOG_LABEL}]", vertical_label=QUERY_TIME_LABEL,
        bar_colors=CANONICAL_BASELINE_COLORS)
    plot.savefig(f"{directory}/plots-{key_size}b/12-constant-selectivity-vs-lookup-time.{file_ext}")

    # === SKEW ===
    # vary uniformity / measure lookup time
    criteria = remove_keys(POINT_QUERY_DEFAULTS, "i_build_key_uniformity", "i_sort_probe")
    plot = bar_plot(
        prefilter(full_pq, criteria), ["i_sort_probe_desc_short", "i_key_concentration"], "probe_time_ms", "i_type", y_stack="sort_time_ms",
        width=7,
        group_label_separator=" / ", stack_label="sort", horizontal_label=f"Query order / Key concentration", vertical_label=QUERY_TIME_LABEL,
        bar_colors=CANONICAL_BASELINE_COLORS)
    plot.savefig(f"{directory}/plots-{key_size}b/13-build-skew-vs-lookup-time.{file_ext}")

    # === SKEW ===
    # vary zipf parameter for queries / measure lookup time
    criteria = remove_keys(POINT_QUERY_DEFAULTS, "i_probe_zipf_coefficient", "i_sort_probe")
    plot = bar_plot(
        prefilter(full_pq, criteria), ["i_sort_probe_desc_short", "i_probe_zipf_coefficient"], "probe_time_ms", "i_type", y_stack="sort_time_ms",
        width=7,
        group_label_separator=" / ", stack_label="sort", horizontal_label=f"Query order / Zipf coefficient for queries", vertical_label=QUERY_TIME_LABEL,
        bar_colors=CANONICAL_BASELINE_COLORS)
    plot.savefig(f"{directory}/plots-{key_size}b/14-query-skew-vs-lookup-time.{file_ext}")

    # ===================== UNUSED PLOTS =====================

    # vary range query size / measure time
    criteria = RANGE_QUERY_DEFAULTS | {"i_log_key_range_factor": 0}
    plot = bar_plot(
        prefilter(full_rq, criteria), "i_log_range_query_size", "probe_time_per_hit", "i_type",
        legend=True,
        horizontal_label=f"Number of qualifying entries [{LOG_LABEL}]", vertical_label="Norm. cum. query time [ms]",
        bar_colors=CANONICAL_BASELINE_COLORS)
    plot.savefig(f"{directory}/plots-{key_size}b/x-range-query-size-vs-lookup-time-per-hit.{file_ext}")

    # unsorted build / vary lookup order / measure time
    criteria = remove_keys(POINT_QUERY_DEFAULTS, "i_sort_probe")
    plot = bar_plot(
        prefilter(full_pq, criteria), "i_sort_probe_desc", "probe_time_ms", "i_type", y_stack="sort_time_ms",
        stack_label="sort", horizontal_label="", vertical_label=QUERY_TIME_LABEL,
        bar_colors=CANONICAL_BASELINE_COLORS)
    plot.savefig(f"{directory}/plots-{key_size}b/x-lookup-order.{file_ext}")

    # sorted lookups / vary build order / measure time
    criteria = remove_keys(POINT_QUERY_DEFAULTS, "i_sort_probe", "i_sort_insert") | {"i_sort_probe": 1}
    plot = bar_plot(
        prefilter(full_pq, criteria), "i_sort_insert_desc", "probe_time_ms", "i_type", y_stack="sort_time_ms",
        stack_label="sort", horizontal_label="", vertical_label=QUERY_TIME_LABEL,
        bar_colors=CANONICAL_BASELINE_COLORS)
    plot.savefig(f"{directory}/plots-{key_size}b/x-build-order-with-sorted-lookup.{file_ext}")

    # unsorted lookups / vary build order / measure time
    criteria = remove_keys(POINT_QUERY_DEFAULTS, "i_sort_probe", "i_sort_insert") | {"i_sort_probe": 0}
    plot = bar_plot(
        prefilter(full_pq, criteria), "i_sort_insert_desc", "probe_time_ms", "i_type", y_stack="sort_time_ms",
        stack_label="sort", horizontal_label="", vertical_label=QUERY_TIME_LABEL,
        bar_colors=CANONICAL_BASELINE_COLORS)
    plot.savefig(f"{directory}/plots-{key_size}b/x-build-order-with-unsorted-lookup.{file_ext}")

    # sorted lookups / vary hit rate / measure lookup time
    criteria = remove_keys(POINT_QUERY_DEFAULTS, "i_hit_rate", "i_sort_probe") | {"i_sort_probe": 1}
    plot = bar_plot(
        prefilter(full_pq, criteria), "i_hit_rate", "probe_time_ms", "i_type", y_stack="sort_time_ms",
        legend_separator="/", stack_label="sort", horizontal_label="hit rate", vertical_label=QUERY_TIME_LABEL,
        bar_colors=CANONICAL_BASELINE_COLORS)
    plot.savefig(f"{directory}/plots-{key_size}b/x-hit-rate-vs-lookup-time-sorted.{file_ext}")

    # unsorted lookups / vary hit rate / measure lookup time
    criteria = remove_keys(POINT_QUERY_DEFAULTS, "i_hit_rate", "i_sort_probe") | {"i_sort_probe": 0}
    plot = bar_plot(
        prefilter(full_pq, criteria), "i_hit_rate", "probe_time_ms", "i_type", y_stack="sort_time_ms",
        legend_separator="/", stack_label="sort", horizontal_label="hit rate", vertical_label=QUERY_TIME_LABEL,
        bar_colors=CANONICAL_BASELINE_COLORS)
    plot.savefig(f"{directory}/plots-{key_size}b/x-hit-rate-vs-lookup-time-unsorted.{file_ext}")

    # sorted queries / vary range query size / measure lookup time
    criteria = remove_keys(RANGE_QUERY_DEFAULTS, "i_sort_probe") | {"i_log_key_range_factor": 0, "i_sort_probe": 1}
    plot = bar_plot(
        prefilter(full_rq, criteria), "i_log_range_query_size", "probe_time_ms", "i_type", y_stack="sort_time_ms",
        log_scale=True, lower_limit=10, upper_limit=100_000,
        horizontal_label=f"Number of qualifying entries [{LOG_LABEL}]", vertical_label=QUERY_TIME_LABEL,
        bar_colors=CANONICAL_BASELINE_COLORS)
    plot.savefig(f"{directory}/plots-{key_size}b/x-range-query-size-vs-lookup-time-sorted.{file_ext}")

    # sorted queries / vary key range / measure lookup time
    criteria = remove_keys(RANGE_QUERY_DEFAULTS, "i_sort_probe") | {"i_log_range_query_size": 10, "i_sort_probe": 1}
    plot = bar_plot(
        prefilter(full_rq, criteria), "i_log_key_range", "probe_time_ms", "i_type", y_stack="sort_time_ms",
        log_scale=True, lower_limit=10, upper_limit=100_000,
        horizontal_label=f"Maximum key [{LOG_LABEL}]", vertical_label=QUERY_TIME_LABEL,
        bar_colors=CANONICAL_BASELINE_COLORS)
    plot.savefig(f"{directory}/plots-{key_size}b/x-key-range-vs-lookup-time-sorted.{file_ext}")

    # sorted queries / keep hits constant / vary key range / measure lookup time
    criteria = remove_keys(RANGE_QUERY_DEFAULTS, "i_sort_probe") | {"i_log_range_query_hits": 2, "i_sort_probe": 1}
    plot = bar_plot(
        prefilter(full_rq, criteria), ["i_log_range_query_size", "i_log_key_range"], "probe_time_ms", "i_type", y_stack="sort_time_ms",
        width=7,
        group_label_separator=" / ", horizontal_label=f"Range query span (s) [{LOG_LABEL}] / Maximum key [{LOG_LABEL}]", vertical_label=QUERY_TIME_LABEL,
        bar_colors=CANONICAL_BASELINE_COLORS)
    plot.savefig(f"{directory}/plots-{key_size}b/x-constant-selectivity-vs-lookup-time-sorted.{file_ext}")


def plot_key_size_comparison_experiments(file_ext, directory):
    full_pq = pd.concat([load_csv_with_heading(f"{directory}/point_query_k{ks}_v32.csv") for ks in [32, 64]])
    full_pq = full_pq.replace({"i_key_size": {32: "32-bit keys", 64: "64-bit keys"}})
    full_pq["gpu_resident_gb"] = full_pq["gpu_resident_bytes"] / 1_000_000_000
    full_pq["i_type"] = full_pq["i_type"].replace(CANONICAL_BASELINE_NAMES)
    full_pq = enforce_value_order(full_pq, "i_type", CANONICAL_BASELINE_ORDER)

    os.makedirs(f"{directory}/plots-all", exist_ok=True)

    # === KEY SIZE ===
    # lookup time
    plot = bar_plot(
        prefilter(full_pq, POINT_QUERY_DEFAULTS), None, "probe_time_ms", "i_type", "i_key_size",
        decomposed_legend=True,
        horizontal_label="", vertical_label=QUERY_TIME_LABEL,
        bar_colors=CANONICAL_BASELINE_COLORS)
    plot.savefig(f"{directory}/plots-all/key-size-vs-probe-time.{file_ext}")

    # === KEY SIZE ===
    # build time
    plot = bar_plot(
        prefilter(full_pq, POINT_QUERY_DEFAULTS), None, "build_time_ms", "i_type", "i_key_size",
        legend=False,
        decomposed_legend=True,
        horizontal_label="", vertical_label=BUILD_TIME_LABEL,
        bar_colors=CANONICAL_BASELINE_COLORS)
    plot.savefig(f"{directory}/plots-all/key-size-vs-build-time.{file_ext}")

    # === KEY SIZE ===
    # build size
    plot = bar_plot(
        prefilter(full_pq, POINT_QUERY_DEFAULTS), None, "gpu_resident_gb", "i_type", "i_key_size",
        legend=False,
        decomposed_legend=True,
        horizontal_label="", vertical_label=MEMORY_FOOTPRINT_LABEL,
        bar_colors=CANONICAL_BASELINE_COLORS)
    plot.savefig(f"{directory}/plots-all/key-size-vs-index-size.{file_ext}")


def plot_hardware_experiments(file_ext, directories, key_size):
    full_pq = pd.concat([load_csv_with_heading(f"{d}/point_query_k{key_size}_v32.csv").assign(i_path=d) for d in directories])
    full_pq["i_type"] = full_pq["i_type"].replace(CANONICAL_BASELINE_NAMES)
    full_pq["i_path"] = full_pq["i_path"].replace({"2080ti": "2080Ti", "a6000": "A6000"})
    full_pq["i_sort_probe_desc_short"] = full_pq["i_sort_probe"].replace({0: "u", 1: "s"})
    full_pq = enforce_value_order(full_pq, "i_path", ["4090", "A6000", "3090", "2080Ti"])
    full_pq = enforce_value_order(full_pq, "i_type", CANONICAL_BASELINE_ORDER)
    full_pq = enforce_value_order(full_pq, "i_sort_probe", [1, 0])

    criteria = remove_keys(POINT_QUERY_DEFAULTS, "i_sort_probe")

    print(prefilter(full_pq, criteria | {"i_path": ["4090", "2080Ti"], "i_sort_probe": 0})[["i_key_size", "i_path", "i_type", "probe_time_ms"]].to_string())

    plot = bar_plot(
        prefilter(full_pq, criteria), ["i_sort_probe_desc_short", "i_path"], "probe_time_ms", "i_type", y_stack="sort_time_ms",
        sort_groups=False,
        width=7,
        group_label_separator=" / ", stack_label="sort", horizontal_label="Query order / Hardware setup", vertical_label=QUERY_TIME_LABEL,
        bar_colors=CANONICAL_BASELINE_COLORS)
    plot.savefig(f"hardware-{key_size}b-lookups-vs-lookup-time.{file_ext}")


def plot_all(file_ext):
    subdirectories = [d for d in os.listdir(".") if os.path.isdir(d)]

    for d in subdirectories:
        if os.path.exists(f"{d}/experiments"):
            plot_design_experiments(file_ext, d)
        plot_comparison_experiments(file_ext, d, 32)
        plot_comparison_experiments(file_ext, d, 64)
        plot_key_size_comparison_experiments(file_ext, d)
    
    plot_hardware_experiments(file_ext, subdirectories, 32)
    plot_hardware_experiments(file_ext, subdirectories, 64)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    #plot_all("png")
    plot_all("pdf")


if __name__ == "__main__":
    main()
