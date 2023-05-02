# visualize/theme.py

"""Provides visualization themes for plotting phi structures."""

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Union

# TODO convert to nested structure?


@dataclass(kw_only=True)
class Theme:
    """Specifies plot attributes."""

    fontfamily: str = "MesloLGS NF, Roboto Mono, Menlo"
    fontsize: int = 12
    direction_offset: float = 0.5
    purview_shape: Union[str, Callable] = "log_n_choose_k"
    purview_offset_radius: float = 0.1
    cause_color: str = "#e21a1a"
    effect_color: str = "#14b738"
    point_size_range: tuple = (5, 30)
    distinction: bool = True
    distinction_mode: str = "text+markers"
    distinction_opacity: float = 0.75
    distinction_colorscale: str = "viridis"
    distinction_color_range: tuple[float] = (0, 0.8)
    distinction_opacity_range: tuple = (0.1, 0.9)
    line_width_range: tuple = (3, 10)
    cause_effect_link: bool = True
    cause_effect_link_color: str = "lightgrey"
    cause_effect_link_opacity: float = 0.5
    mechanism_purview_link: bool = True
    mechanism_purview_link_color: str = "lightgrey"
    mechanism_purview_link_opacity: float = 0.5
    mechanism: bool = True
    mechanism_max_radius: float = 1.0
    mechanism_z_offset: float = 0.0
    mechanism_z_spacing: float = 0.0
    mechanism_shape: Union[Callable, str] = "linear"
    purview_radius_mod: float = 1.0
    """Controls whether a single trace is used to plot 2-relation faces,
    precluding visual indications of their phi value."""
    two_relation: bool = True
    two_relation_detail_threshold: int = 1000
    two_relation_opacity: float = 0.1
    two_relation_line_width: float = 1
    two_relation_color: str = None
    two_relation_colorscale: Union[str, Callable, Mapping] = "type"
    two_relation_showscale: bool = True
    two_relation_reversescale: bool = False
    two_relation_hoverlabel_font_color: str = "white"
    three_relation: bool = True
    three_relation_colorscale: str = "teal"
    three_relation_reversescale: bool = False
    three_relation_showscale: bool = True
    three_relation_opacity: float = 0.1
    three_relation_opacity_range: Optional[tuple] = None
    three_relation_intensity_range: tuple = (0, 1)
    three_relation_showlegend: bool = True
    lighting: Mapping = field(
        default_factory=lambda: dict(
            ambient=0.8, diffuse=0, roughness=0, specular=0, fresnel=0
        )
    )
    legendgroup_postfix: str = ""


@dataclass(kw_only=True)
class Grey(Theme):
    """A grey theme."""

    cause_color: str = "grey"
    effect_color: str = "grey"
    distinction_colorscale: str = "greys"
    distinction_opacity_range: tuple[float] = (0.1, 0.2)
    cause_effect_link_color = "grey"
    cause_effect_link_opacity: float = 0.1
    mechanism_purview_link_color = "grey"
    mechanism_purview_link_opacity: float = 0.1
    two_relation_colorscale: str = "greys"
    two_relation_opacity: float = 0.1
    two_relation_showscale: bool = False
    three_relation_colorscale: str = "greys"
    three_relation_opacity: float = 0.05
    three_relation_intensity_range: tuple[float] = (0, 0.5)
    legendgroup_postfix: str = " (greyed)"
