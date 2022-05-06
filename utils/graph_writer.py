from typing import List, Dict, Any, Optional, Tuple
import plotly.graph_objects as go
from plotly.colors import qualitative
import numpy as np
from models.base import ValueModel
from itertools import cycle
from utils.cache import AttributeValueGaussianCacheEntry
import torch
import math


DimensionScatterGraphData = List[Tuple[AttributeValueGaussianCacheEntry, List[Dict[str, Any]]]]

QUALITATIVE_COLORS = qualitative.Set2
QUALITATIVE_COLORS_SCATTER = qualitative.Dark2


class GraphWriter:
    def __init__(self, selected_results: List[Dict[str, Any]]):
        self._selected_results = selected_results

    @staticmethod
    def plot_dimension_scatter_graph(groups: DimensionScatterGraphData, dim_1: int, dim_2: int, device, show_legend: bool = False, specification: Optional[List[Dict[str, Any]]] = None):
        fig = go.Figure()
        for idx, (color, (cache, points)) in enumerate(zip(cycle(QUALITATIVE_COLORS_SCATTER), groups)):
            name = cache.get_value()

            # Plot points
            word = [p["word"] for p in points]
            x_point = [p["coordinate"][0] for p in points]
            y_point = [p["coordinate"][1] for p in points]

            if "log_prob" in points[0]:
                log_prob = [p["log_prob"] for p in points]
                prob = [math.exp(p["log_prob"]) for p in points]
                custom_data = list(zip(word, log_prob, prob))
                hover_template = '<b>%{customdata[0]}</b><br>log_prob: %{customdata[1]:.3f} \
                    <br>prob: %{customdata[2]:.3f}'
            else:
                custom_data = word
                hover_template = '<b>%{customdata}</b>'

            fig.add_trace(go.Scatter(
                x=x_point, y=y_point, mode="markers", name=name,
                customdata=custom_data,
                hovertemplate=hover_template,
                marker_color=color, marker=dict(opacity=0.5),
                showlegend=show_legend,
            ))

        if not specification:
            for idx, (color, (cache, points)) in enumerate(zip(cycle(QUALITATIVE_COLORS_SCATTER), groups)):
                name = cache.get_value()
                darkening_factor = 0.75

                # Darken contour
                rgb_ints = [str(int(x.strip(" ")) * darkening_factor) for x in color[4:][:-1].split(",")]
                new_color = f"rgb({','.join(rgb_ints)})"

                # Plot contours
                num_contours = 3
                num_points = 100
                for r in range(1, num_contours):
                    # Compute circle points
                    angles = torch.linspace(0, 2 * np.pi, steps=num_points).to(device)
                    x = r * angles.cos()
                    y = r * angles.sin()
                    vectors = torch.stack([x, y], dim=1)

                    # Transform according to model
                    dims = [dim_1, dim_2]
                    mean = cache.get_gaussian_model_params()[0].to(device)[dims]
                    cov = cache.get_gaussian_model_params()[1].to(device)[dims].t()[dims].t()
                    cov_lt = torch.cholesky(cov)
                    vectors = mean.unsqueeze(0) + vectors.matmul(cov_lt.t())
                    vals = [l.squeeze().tolist() for l in vectors.cpu().split(dim=1, split_size=1)]
                    x, y = vals[0], vals[1]

                    # Plot
                    contour_data = {"x": x, "y": y, "mode": "lines", "marker_color": new_color,
                                    "legendgroup": f"{name} (Model)", "name": f"{name} (Model)"}
                    if r == num_contours - 1:
                        fig.add_trace(go.Scatter(showlegend=show_legend, **contour_data))
                    else:
                        fig.add_trace(go.Scatter(showlegend=False, **contour_data))
        else:
            # Plot from specification
            for idx, (color, params, (cache, _)) in enumerate(zip(cycle(QUALITATIVE_COLORS_SCATTER), specification, groups)):
                name = cache.get_value()
                darkening_factor = 0.75

                # Darken contour
                rgb_ints = [str(int(x.strip(" ")) * darkening_factor) for x in color[4:][:-1].split(",")]
                new_color = f"rgb({','.join(rgb_ints)})"

                # Plot Gaussian contours
                num_contours = 3
                num_points = 100
                for r in range(1, num_contours):
                    # Compute circle points
                    angles = torch.linspace(0, 2 * np.pi, steps=num_points).to(device)
                    x = r * angles.cos()
                    y = r * angles.sin()
                    vectors = torch.stack([x, y], dim=1)

                    # Transform according specification
                    dims = [dim_1, dim_2]
                    mean = params["mean_n"].to(device)[dims]

                    if "std_n" in params:
                        cov = params["std_n"].to(device)[dims].pow(2).diag()
                    elif "cov_n" in params:
                        cov = params["cov_n"].to(device)[dims].t()[dims].t()

                    cov_lt = torch.cholesky(cov)
                    gaussian_vectors = mean.unsqueeze(0) + vectors.matmul(cov_lt.t())
                    vals = [l.squeeze().tolist() for l in gaussian_vectors.cpu().split(dim=1, split_size=1)]
                    x_gaussian, y_gaussian = vals[0], vals[1]

                    cauchy_loc = params["cauchy_loc_n"].to(device)[dims]
                    cauchy_cov = params["cauchy_scale_n"].float().to(device)[dims].pow(2).diag()
                    cauchy_cov_lt = torch.cholesky(cauchy_cov)
                    bernoulli_prob = params["bernoulli_prob_n"].to(device)[dims].tolist()
                    bernoulli_prob_string = "/".join([f"{x:.3f}" for x in bernoulli_prob])

                    cauchy_vectors = cauchy_loc.unsqueeze(0) + vectors.matmul(cauchy_cov_lt.t())
                    cauchy_vals = [l.squeeze().tolist() for l in cauchy_vectors.cpu().split(dim=1, split_size=1)]
                    x_cauchy, y_cauchy = cauchy_vals[0], cauchy_vals[1]

                    # Plot Gaussian
                    contour_data = {
                        "x": x_gaussian, "y": y_gaussian, "mode": "lines", "marker_color": new_color,
                        "legendgroup": f"{name} (Gaussian)", "name": f"{name} (Gaussian)"}

                    cauchy_contour_data = {
                        "x": x_cauchy, "y": y_cauchy, "mode": "lines", "marker_color": new_color,
                        "legendgroup": f"{name} (Cauchy)", "name": f"{name} (Cauchy; {bernoulli_prob_string})", "line": {"dash": "dot"}}

                    if r == num_contours - 1:
                        fig.add_trace(go.Scatter(showlegend=show_legend, **cauchy_contour_data))
                        fig.add_trace(go.Scatter(showlegend=show_legend, **contour_data))
                    else:
                        fig.add_trace(go.Scatter(showlegend=False, **cauchy_contour_data))
                        fig.add_trace(go.Scatter(showlegend=False, **contour_data))


        fig.update_xaxes(title_text=f"Dimension {dim_1}")
        fig.update_yaxes(title_text=f"Dimension {dim_2}")

        # Make serif
        fig.update_layout(
            font=dict(family="serif"),
            margin=dict(l=0, r=0, t=0, b=0),
        )

        return fig

    @staticmethod
    def plot_dimensions_graph(
            fig: go.Figure, y: List[float], y_labels: Optional[List[str]] = None, y_err: Optional[List[float]] = None,
            practical_maximum: Optional[float] = None, label: Optional[str] = None):
        color = QUALITATIVE_COLORS[0]
        rgb_ints = [str(int(x.strip(" "))) for x in color[4:][:-1].split(",")]
        new_color = f"rgba({','.join(rgb_ints + ['0.2'])})"

        # Error range
        x = list(range(1, len(y) + 1))
        if y_err:
            y_upper = [m + e for m, e in zip(y, y_err)]
            y_lower = [m - e for m, e in zip(y, y_err)]
            fig.add_trace(go.Scatter(
                x=x + x[::-1],
                y=y_upper + y_lower[::-1],
                fill='toself',
                fillcolor=new_color,
                line_color='rgba(255,255,255,0)',
                showlegend=False,
                name=label,
                hoverinfo='skip',
                legendgroup=label,
            ))

        custom_data: Optional[List[Any]] = None
        hover_template = None
        if y_labels:
            if y_err:
                custom_data = list(zip(y_labels, y_err))
                hover_template = '<b>%{y:.3f} +- %{customdata[1]:.3f}</b><br>%{customdata[0]}'
            else:
                custom_data = y_labels
                hover_template = '<b>%{y:.3f}</b><br>%{customdata}'

        # Mean line
        fig.add_trace(go.Scatter(
            x=x, y=y,
            customdata=custom_data,
            hovertemplate=hover_template,
            line_color=color,
            showlegend=True,
            name=label,
            legendgroup=label,
        ))
        fig.update_traces(mode='lines')

        # Max MI
        if practical_maximum:
            fig.add_shape(
                # Line Horizontal
                type="line",
                x0=min(x),
                y0=practical_maximum,
                x1=max(x),
                y1=practical_maximum,
                line=dict(
                    color="LightSeaGreen",
                    dash="dash",
                ),
            )

        fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))  # noqa
        fig.update_xaxes(title_text="Dimensions Selected", tickvals=x)

        # Make serif
        fig.update_layout(font=dict(
            family="serif",
        ))

        return fig

    @staticmethod
    def add_to_plot(
            fig: go.Figure, y: List[float], y_labels: Optional[List[str]] = None, y_err: Optional[List[float]] = None,
            practical_maximum: Optional[float] = None, label: Optional[str] = None, index: int = 0):
        color = QUALITATIVE_COLORS[index % len(QUALITATIVE_COLORS)]
        rgb_ints = [str(int(x.strip(" "))) for x in color[4:][:-1].split(",")]
        new_color = f"rgba({','.join(rgb_ints + ['0.2'])})"

        custom_data: Optional[List[Any]] = None
        hover_template = None
        if y_labels:
            if y_err:
                custom_data: List[Any] = list(zip(y_labels, y_err))
                hover_template = '<b>%{y:.3f} +- %{customdata[1]:.3f}</b><br>%{customdata[0]}'
            else:
                custom_data = y_labels
                hover_template = '<b>%{y:.3f}</b><br>%{customdata}'

        # Error range
        x = list(range(1, len(y) + 1))
        if y_err:
            y_upper = [m + e for m, e in zip(y, y_err)]
            y_lower = [m - e for m, e in zip(y, y_err)]
            fig.add_trace(go.Scatter(
                x=x + x[::-1],
                y=y_upper + y_lower[::-1],
                fill='toself',
                fillcolor=new_color,
                line_color='rgba(255,255,255,0)',
                showlegend=False,
                name=label,
                hoverinfo='skip',
                legendgroup=label,
            ))

        # Mean line
        fig.add_trace(go.Scatter(
            x=x, y=y,
            customdata=custom_data,
            hovertemplate=hover_template,
            line_color=color,
            showlegend=True,
            name=label,
            legendgroup=label,
        ))
        fig.update_traces(mode='lines')

        # Make serif
        fig.update_layout(font=dict(
            family="serif",
        ))

        return fig

    def plot_mi(self, theoretical_maximum: float, practical_maximum: Optional[float] = None):
        # Construct Y-value arrays
        y = [x["mi"].nominal_value for x in self._selected_results]
        y_err = [x["mi"].std_dev for x in self._selected_results]
        y_labels = [", ".join([str(i) for i in x["candidate_dim_pool"]]) for x in self._selected_results]

        fig = go.Figure()
        fig = GraphWriter.plot_dimensions_graph(
            fig=fig, y=y, y_err=y_err, y_labels=y_labels, practical_maximum=practical_maximum,
            label="Mutual Information")

        # Add entropy line
        y_entropy = [x["entropy"] for x in self._selected_results]
        fig = GraphWriter.add_to_plot(
            fig=fig, y=y_entropy, y_labels=y_labels, label="Entropy")

        # Add conditional entropy line
        y_entropy = [x["conditional_entropy"].nominal_value for x in self._selected_results]
        y_err_entropy = [x["conditional_entropy"].std_dev for x in self._selected_results]
        fig = GraphWriter.add_to_plot(
            fig=fig, y=y_entropy, y_err=y_err_entropy, y_labels=y_labels, label="Conditional Entropy")

        # Setup axes and labels
        fig.update_yaxes(title_text="Mutual Information", range=[0.0, theoretical_maximum + 0.05])

        return fig

    def plot_normalized_mi(self, theoretical_maximum: float, practical_maximum: Optional[float] = None):
        # Construct Y-value arrays
        y = [x["mi"].nominal_value / theoretical_maximum for x in self._selected_results]
        y_err = [x["mi"].std_dev / theoretical_maximum for x in self._selected_results]
        y_labels = [", ".join([str(i) for i in x["candidate_dim_pool"]]) for x in self._selected_results]

        fig = go.Figure()
        fig = GraphWriter.plot_dimensions_graph(
            fig=fig, y=y, y_err=y_err, y_labels=y_labels, practical_maximum=practical_maximum,
            label="Normalized Mutual Information")

        # Setup axes and labels
        fig.update_yaxes(title_text="Normalized Mutual Information", range=[0.0, 1.0])

        return fig

    def plot_accuracy(self, practical_maximum: Optional[float] = None):
        # Construct Y-value arrays
        y = [x["model_accuracy"] for x in self._selected_results]
        y_labels = [", ".join([str(i) for i in x["candidate_dim_pool"]]) for x in self._selected_results]

        fig = go.Figure()
        fig = GraphWriter.plot_dimensions_graph(
            fig=fig, y=y, y_labels=y_labels, practical_maximum=practical_maximum, label="Model")

        # Majority class baseline
        y_baseline = [x["baseline_accuracy"] for x in self._selected_results]
        fig = GraphWriter.add_to_plot(
            fig=fig, y=y_baseline, y_labels=y_labels, label="Majority-class Baseline")

        # Setup y-axis
        fig.update_yaxes(title_text="Accuracy", range=[0.0, 1.0])

        return fig
