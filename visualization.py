"""Visualization functions for ULM density and velocity maps.

Author: Tristan Stevens
"""

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

from read import Config, ReadTracks, load_config_from_yaml
from utils import translate, yellow


def imagesc(x, y, data, ax=None, **kwargs):
    """Imagesc function from Matlab implemented using Matplotlib."""

    def extents(f):
        delta = f[1] - f[0]
        return [f[0] - delta / 2, f[-1] + delta / 2]

    if ax is None:
        ax = plt

    im = ax.imshow(
        data,
        aspect=1,
        interpolation="none",
        extent=extents(x) + extents(y)[::-1],
        origin="upper",
        **kwargs,
    )

    return im


def matplotlib_to_pil(image):
    """Matplotlib image to PIL image."""
    pil_image = Image.fromarray(np.array(image * 255).astype(np.uint8))
    return pil_image


def add_colorbar(fig, ax, im):
    """Add a colorbar to a figure axis."""
    fig.subplots_adjust(right=0.85)
    cax = fig.add_axes(
        [
            ax.get_position().x1 + 0.01,
            ax.get_position().y0,
            0.02,
            ax.get_position().height,
        ]
    )
    fig.colorbar(im, cax=cax)
    return fig, cax


def get_axial_velocity_map():
    """Create a colormap for axial velocity maps."""
    hot1 = matplotlib.colormaps["hot"](np.linspace(0, 1, 128))
    hot2 = matplotlib.colormaps["hot"](np.linspace(0, 1, 128))
    hot2 = np.flip(np.flip(hot2, axis=0), axis=1)
    hot2 = hot2[:, [1, 2, 3, 0]]
    # combine them and build a new colormap
    colors = np.vstack((hot2, hot1))
    # remove white parts
    edge = 5
    colors = colors[edge : -1 - edge, :]
    cmapvz = mcolors.LinearSegmentedColormap.from_list("velocity_map", colors)
    return cmapvz


def ind2rgb(indices, cmap):
    """Ind2rgb function from Matlab implemented using Matplotlib."""
    if isinstance(cmap, matplotlib.colors.LinearSegmentedColormap):
        cmap = np.array([cmap(i) for i in range(cmap.N)])

    image = cmap[indices, :]
    return image


def image_from_colormap(image, cmap):
    """Convert image to RGB image using the given colormap."""
    if np.min(image) < 0:
        image = translate(image, (-np.abs(image).max(), np.abs(image).max()), (0, 1))
    else:
        image = translate(image, (0, np.abs(image).max()), (0, 1))

    indx = np.array(image * 255).astype(np.uint8)
    image_rgb = ind2rgb(indx, cmap)[..., :3]
    return image_rgb


def brighten_map(cmap, beta):
    """Brighten colormap by raising it to the power of gamma."""
    tol = np.sqrt(sys.float_info.epsilon)
    if beta > 0:
        gamma = 1 - min(1 - tol, beta)
    else:
        gamma = 1 / (1 + max(-1 + tol, beta))

    return cmap**gamma


def show_ulm(
    ulm_den=None,
    ulm_vz=None,
    ulm_vx=None,
    ulm_v=None,
    saturation: float = 0.8,
    sigma_gauss: float = 0.1,
    brighten: float = 0.4,
    shadow: float = 0.3,
    comprVel: float = 0.25,
    comprDen: float = 0.33,
    save_dir: str = None,
    figsize: float = 6,
    filetype: str = "png",
    dpi: int = 600,
    title: str = None,
    axis: bool = True,
    colorbar: bool = True,
    style: str = "ius",
    show_matplotlib: bool = False,
    show_pillow: bool = False,
    black_background: bool = True,
    optimal_settings: dict = None,
    x_axis: np.ndarray = None,
    z_axis: np.ndarray = None,
    tag: str = None,
):
    """Plot ULM density / velocity maps.

    Saves both matplotlib plot (with axis and colorbar) as well as the
    raw PIL image in original image grid.

    Args:
        ulm_den (np.ndarray, optional): ULM density map. Defaults to None.
        ulm_vz (np.ndarray, optional): ULM axial velocity in mm/s. Defaults to None.
        ulm_vx (np.ndarray, optional): ULM lateral velocity in mm/s. Defaults to None.
        ulm_v (np.ndarray, optional): ULM velocity magnitude in mm/s. Defaults to None.
        saturation (float, optional): saturation of the image. Defaults to 0.8.
        sigma_gauss (float, optional): sigma of gaussian filter. Defaults to 0.1.
        brighten (float, optional): brighten value. Defaults to 0.4.
        shadow (float, optional): shadow value. Defaults to 0.3.
        comprVel (float, optional): compression value for velocity. Defaults to 0.25.
        comprDen (float, optional): compression value for density. Defaults to 0.33.
        save_dir (str, optional): save directory. Defaults to None.
        figsize (float, optional): figure size (width) in inches. Aspect ratio
            is automatically detected and height is set accordingly. Defaults to 6.
        filetype (str, optional): save to`jpg`, png` or `pdf`. Defaults to 'jpg'.
        dpi (int, optional): dots per inch of saved image. Defaults to 600.
        title (str, optional): title of matplotlib plot. Defaults to None.
        axis (bool, optional): whether to include axis to matplotlib plot.
            Defaults to True.
        colorbar (bool, optional): Whether to include colorbar to matplotlib plot.
            Defaults to True.
        style (str, optional): Plot in style of ius challenge or similar to pala.
            Defaults to 'ius'.
        show_matplotlib (bool, optional): Whether to show matplotlib plot or only
            final PIL image. Defaults to False.
        show_pillow (bool, optional): Whether to show PIL image. Defaults to False.
        x_axis (np.ndarray, optional): x-axis values. Defaults to None.
        z_axis (np.ndarray, optional): z-axis values. Defaults to None.
        tag (str, optional): tag to add to the saved image. Defaults to None.
    """

    assert filetype in ["pdf", "png", "jpg"]
    assert style in ["ius", "pala"]

    output_images = {}
    fig = None

    cmapvz = get_axial_velocity_map()
    cmapvx = cmapvz
    cmapv = matplotlib.colormaps["jet"]
    if style == "pala":
        cmapden = matplotlib.colormaps["gray"]
    elif style == "ius":
        cmapden = matplotlib.colormaps["hot"]

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    size = figsize

    if show_matplotlib:
        assert x_axis is not None, "x_axis should be provided for matplotlib plot"
        assert z_axis is not None, "z_axis should be provided for matplotlib plot"

    def save_image(image, fig, file_path, tag=None):
        """Saves both PIL image and matplotlib image

        Args:
            image (PIL Image): PIL image object
            fig (matplotilb.figure): matplotlib figure object
            file_path (str): path where to save images
        Returns:
            will save both
                f{file_path.stem}.png and
                f{file_path.stem}_plot.png
        """
        if tag is not None:
            file_path = file_path.parent / f"{file_path.stem}_{tag}.{filetype}"

        image.save(file_path)
        if tag is None:
            plot_file_path = file_path.parent / f"{file_path.stem}_plot.{filetype}"
        else:
            plot_file_path = (
                file_path.parent / f"{file_path.stem}_{tag}_plot.{filetype}"
            )

        if show_matplotlib:
            if axis:
                fig.savefig(plot_file_path, dpi=dpi, bbox_inches="tight")
            else:
                plt.subplots_adjust(
                    left=0, bottom=0, right=1, top=1, wspace=0, hspace=0
                )
                fig.savefig(plot_file_path, dpi=dpi)

        print(f"Succesfully saved image to {yellow(file_path)}")

    if ulm_den is not None:
        ulm_v_shadow = ulm_den / np.max(ulm_den * shadow)
        ulm_v_shadow[ulm_v_shadow > 1] = 1
    elif ulm_v is not None:
        ulm_v_shadow = ulm_v / np.max(ulm_v * shadow)
        ulm_v_shadow[ulm_v_shadow > 1] = 1
    else:
        ulm_v_shadow = 1

    ## ULM - Density Map
    if ulm_den is not None:
        if optimal_settings is not None:
            if "ulm_den" in optimal_settings:
                saturation = optimal_settings.ulm_den.saturation
                comprDen = optimal_settings.ulm_den.comprDen
                sigma_gauss = optimal_settings.ulm_den.sigma_gauss
            else:
                print("Could not find ulm_den in optimal settings")

        aspect_ratio = np.divide(*ulm_den.shape)
        image = ulm_den**comprDen

        if sigma_gauss:
            image = gaussian_filter(image, sigma=sigma_gauss)

        if saturation:
            vmax = np.max(image) * saturation
            image = np.clip(image, a_min=image.min(), a_max=vmax)

        fig = None
        if show_matplotlib:
            figsize = (size, aspect_ratio * size)
            fig, ax = plt.subplots(figsize=figsize)

            im = imagesc(
                x_axis,
                z_axis,
                image,
                ax,
                cmap=cmapden,
            )

            if title is True:
                ax.set_title("ULM - Density Map")
            else:
                ax.set_title(title)

            if axis:
                ax.set_xlabel("Width (mm)")
                ax.set_ylabel("Depth (mm)")

                if colorbar:
                    fig, cax = add_colorbar(fig, ax, im)

                    cax.set_ylabel("Number of Counts")
                    ticks = np.linspace(np.min(image), np.max(image), 7)
                    cax.set_yticks(ticks)
                    labels = np.round((ticks ** (1 / comprDen))).astype(int)
                    cax.set_yticklabels(labels)
            else:
                ax.axis("off")

        image = image_from_colormap(image, cmapden)
        image = matplotlib_to_pil(image)
        output_images["ulm_den"] = image
        if show_pillow:
            image.show()

        if save_dir:
            file_path = Path(save_dir, f"ulm_den.{filetype}")
            save_image(image, fig, file_path, tag=tag)

    ## ULM - Density Map with Axial Flow Direction
    if (ulm_vz is not None) and (ulm_den is not None):
        if optimal_settings is not None:
            if "ulm_vz_ax_flow" in optimal_settings:
                saturation = optimal_settings.ulm_vz_ax_flow.saturation
                comprVel = optimal_settings.ulm_vz_ax_flow.comprVel
                sigma_gauss = optimal_settings.ulm_vz_ax_flow.sigma_gauss
            else:
                print("Could not find ulm_vz_ax_flow in optimal settings")

        aspect_ratio = np.divide(*ulm_vz.shape)
        figsize = (size, aspect_ratio * size)

        image = ulm_den**comprVel * np.sign(gaussian_filter(ulm_vz, 0.8))
        image -= np.sign(image) / 2
        if sigma_gauss:
            image = gaussian_filter(image, sigma=sigma_gauss)

        if show_matplotlib:
            fig, ax = plt.subplots(figsize=figsize)
            im = imagesc(
                x_axis,
                z_axis,
                image,
                cmap=cmapvz,
                vmax=np.max(image) * saturation,
                vmin=-np.max(image) * saturation,
            )

            if title is True:
                ax.set_title("ULM - Density Map with Axial Flow Direction")
            else:
                ax.set_title(title)

            if axis:
                ax.set_xlabel("Width (mm)")
                ax.set_ylabel("Depth (mm)")

                if colorbar:
                    fig, cax = add_colorbar(fig, ax, im)
                    cax.set_ylabel("Count Intensity")
            else:
                ax.axis("off")

        image = image_from_colormap(image, cmapvz)
        image = matplotlib_to_pil(image)
        output_images["ulm_vz_ax_flow"] = image
        if show_pillow:
            image.show()

        if save_dir:
            file_path = Path(save_dir, f"ulm_denvz.{filetype}")
            save_image(image, fig, file_path, tag=tag)

    ## ULM - Density Map with Lateral Flow Direction
    if (ulm_vx is not None) and (ulm_den is not None):
        if optimal_settings is not None:
            if "ulm_vx_ax_flow" in optimal_settings:
                saturation = optimal_settings.ulm_vx_ax_flow.saturation
                comprVel = optimal_settings.ulm_vx_ax_flow.comprVel
                sigma_gauss = optimal_settings.ulm_vx_ax_flow.sigma_gauss
            else:
                print("Could not find ulm_vx_ax_flow in optimal settings")

        aspect_ratio = np.divide(*ulm_vx.shape)
        figsize = (size, aspect_ratio * size)

        image = ulm_den**comprVel * np.sign(gaussian_filter(ulm_vx, 0.8))
        image -= np.sign(image) / 2
        if sigma_gauss:
            image = gaussian_filter(image, sigma=sigma_gauss)

        if show_matplotlib:
            fig, ax = plt.subplots(figsize=figsize)
            im = imagesc(
                x_axis,
                z_axis,
                image,
                cmap=cmapvx,
                vmax=np.max(image) * saturation,
                vmin=-np.max(image) * saturation,
            )

            if title is True:
                ax.set_title("ULM - Density Map with Axial Flow Direction")
            else:
                ax.set_title(title)

            if axis:
                ax.set_xlabel("Width (mm)")
                ax.set_ylabel("Depth (mm)")

                if colorbar:
                    fig, cax = add_colorbar(fig, ax, im)
                    cax.set_ylabel("Count Intensity")
            else:
                ax.axis("off")

        image = image_from_colormap(image, cmapvx)
        image = matplotlib_to_pil(image)
        output_images["ulm_vx_ax_flow"] = image
        if show_pillow:
            image.show()

        if save_dir:
            file_path = Path(save_dir, f"ulm_denvx.{filetype}")
            save_image(image, fig, file_path, tag=tag)

    if ulm_v is not None:
        if np.abs(ulm_v).max() == 0:
            vmax = 1
        else:
            vmax = np.ceil(np.quantile(ulm_v[np.abs(ulm_v) > 0], 0.98) / 10) * 10
    else:
        vmax = None

    ## ULM - Axial Velocity
    if (ulm_v is not None) and (ulm_vz is not None):
        if optimal_settings is not None:
            if "ulm_vz_ax_vel" in optimal_settings:
                shadow = optimal_settings.ulm_vz_ax_vel.shadow
                brighten = optimal_settings.ulm_vz_ax_vel.brighten
                comprVel = optimal_settings.ulm_vz_ax_vel.comprVel
                sigma_gauss = optimal_settings.ulm_vz_ax_vel.sigma_gauss
            else:
                print("Could not find ulm_vz_ax_vel in optimal settings")

        aspect_ratio = np.divide(*ulm_v.shape)
        figsize = (size, aspect_ratio * size)

        if vmax is None:
            if np.abs(ulm_vz).max() == 0:
                vmax = 1
            else:
                vmax = np.ceil(np.quantile(ulm_v[np.abs(ulm_vz) > 0], 0.98) / 10) * 10

        vzmax = 0.8 * vmax
        ulm_vz_rgb = ulm_vz / vzmax
        ulm_vz_rgb = np.sign(ulm_vz_rgb) * np.abs(ulm_vz_rgb) ** (1 / 1.5)
        ulm_vz_rgb[ulm_vz_rgb > 1] = 1
        ulm_vz_rgb[ulm_vz_rgb < -1] = -1
        if sigma_gauss:
            ulm_vz_rgb = gaussian_filter(ulm_vz_rgb, sigma_gauss)

        indx = np.round(ulm_vz_rgb * 128 + 127).astype(int)
        ulm_vz_rgb = ind2rgb(indx, cmapvz)[..., :3]
        ulm_vz_rgb = ulm_vz_rgb * np.expand_dims((ulm_v_shadow**comprVel), axis=-1)
        ulm_vz_rgb = brighten_map(ulm_vz_rgb, brighten)

        if show_matplotlib:
            fig, ax = plt.subplots(figsize=figsize)
            im = imagesc(
                x_axis,
                z_axis,
                ulm_vz_rgb,
                cmap=cmapvz,
            )

            if title is True:
                ax.set_title("ULM - Axial Velocity")
            else:
                ax.set_title(title)

            if axis:
                ax.set_xlabel("Width (mm)")
                ax.set_ylabel("Depth (mm)")

                if colorbar:
                    fig, cax = add_colorbar(fig, ax, im)
                    y_ticks = cax.get_yticks()
                    n_ticks = len(y_ticks)
                    labels = np.round(
                        [-vzmax + i * 2 * vzmax / (n_ticks - 1) for i in range(n_ticks)]
                    ).astype(int)
                    cax.set_ylabel("Axial Velocity (mm/s)")
                    cax.set_yticks(y_ticks)
                    cax.set_yticklabels(labels)
            else:
                ax.axis("off")

        image = matplotlib_to_pil(ulm_vz_rgb)
        output_images["ulm_vz"] = image
        if show_pillow:
            image.show()

        if save_dir:
            file_path = Path(save_dir, f"ulm_vz.{filetype}")
            save_image(image, fig, file_path, tag=tag)

    ## ULM - Lateral Velocity
    if (ulm_v is not None) and (ulm_vx is not None):
        if optimal_settings is not None:
            if "ulm_vx_lat_vel" in optimal_settings:
                shadow = optimal_settings.ulm_vx_lat_vel.shadow
                brighten = optimal_settings.ulm_vx_lat_vel.brighten
                comprVel = optimal_settings.ulm_vx_lat_vel.comprVel
                sigma_gauss = optimal_settings.ulm_vx_lat_vel.sigma_gauss
            else:
                print("Could not find ulm_vx_lat_vel in optimal settings")

        aspect_ratio = np.divide(*ulm_v.shape)
        figsize = (size, aspect_ratio * size)

        if vmax is None:
            if np.abs(ulm_vz).max() == 0:
                vmax = 1
            else:
                vmax = np.ceil(np.quantile(ulm_v[np.abs(ulm_vx) > 0], 0.98) / 10) * 10

        vxmax = 0.8 * vmax
        ulm_vx_rgb = ulm_vx / vxmax
        ulm_vx_rgb = np.sign(ulm_vx_rgb) * np.abs(ulm_vx_rgb) ** (1 / 1.5)
        ulm_vx_rgb[ulm_vx_rgb > 1] = 1
        ulm_vx_rgb[ulm_vx_rgb < -1] = -1
        if sigma_gauss:
            ulm_vx_rgb = gaussian_filter(ulm_vx_rgb, sigma_gauss)

        indx = np.round(ulm_vx_rgb * 128 + 127).astype(int)
        ulm_vx_rgb = ind2rgb(indx, cmapvz)[..., :3]
        ulm_vx_rgb = ulm_vx_rgb * np.expand_dims((ulm_v_shadow**comprVel), axis=-1)
        ulm_vx_rgb = brighten_map(ulm_vx_rgb, brighten)

        if show_matplotlib:
            fig, ax = plt.subplots(figsize=figsize)
            im = imagesc(
                x_axis,
                z_axis,
                ulm_vx_rgb,
                cmap=cmapvz,
            )

            if title is True:
                ax.set_title("ULM - Axial Velocity")
            else:
                ax.set_title(title)

            if axis:
                ax.set_xlabel("Width (mm)")
                ax.set_ylabel("Depth (mm)")

                if colorbar:
                    fig, cax = add_colorbar(fig, ax, im)
                    y_ticks = cax.get_yticks()
                    n_ticks = len(y_ticks)
                    labels = np.round(
                        [-vzmax + i * 2 * vzmax / (n_ticks - 1) for i in range(n_ticks)]
                    ).astype(int)
                    cax.set_ylabel("Axial Velocity (mm/s)")
                    cax.set_yticks(y_ticks)
                    cax.set_yticklabels(labels)
            else:
                ax.axis("off")

        image = matplotlib_to_pil(ulm_vx_rgb)
        output_images["ulm_vx"] = image
        if show_pillow:
            image.show()

        if save_dir:
            file_path = Path(save_dir, f"ulm_vx.{filetype}")
            save_image(image, fig, file_path, tag=tag)

    ## ULM - Velocity Magnitude
    if ulm_v is not None:
        if optimal_settings is not None:
            if "ulm_v" in optimal_settings:
                shadow = optimal_settings.ulm_v.shadow
                brighten = optimal_settings.ulm_v.brighten
                comprVel = optimal_settings.ulm_v.comprVel
                sigma_gauss = optimal_settings.ulm_v.sigma_gauss
                black_background = optimal_settings.ulm_v.black_background
            else:
                print("Could not find ulm_v in optimal settings")

        aspect_ratio = np.divide(*ulm_v.shape)
        figsize = (size, aspect_ratio * size)

        ulm_v_rgb = ulm_v / vmax
        ulm_v_rgb = ulm_v_rgb ** (1 / 1.5)
        ulm_v_rgb[ulm_v_rgb > 1] = 1

        if style == "ius":
            # Gaussian smoothing
            if sigma_gauss:
                ulm_v_rgb = gaussian_filter(
                    ulm_v_rgb, sigma_gauss, truncate=sigma_gauss * 3
                )
            if sigma_gauss:
                ulm_v_shadow = gaussian_filter(
                    ulm_v_shadow, sigma_gauss, truncate=sigma_gauss * 3
                )
            bg = ulm_v_rgb == 0

            ulm_v_rgb = (
                ulm_v_rgb * ulm_v_shadow**comprVel
            )  # np.expand_dims((ulm_v_shadow**comprVel), axis=-1)
            ulm_v_rgb = brighten_map(ulm_v_rgb, brighten)
            # ulm_v_rgb = exposure.equalize_adapthist(ulm_v_rgb)
            ulm_v_rgb = image_from_colormap(ulm_v_rgb, cmapv)

            if black_background:
                ulm_v_rgb[bg] = [0, 0, 0]

        elif style == "pala":
            if sigma_gauss:
                ulm_v_rgb = gaussian_filter(ulm_v_rgb, sigma_gauss)
            ulm_v_rgb = image_from_colormap(ulm_v_rgb, cmapv)
            ulm_v_rgb = ulm_v_rgb * np.expand_dims((ulm_v_shadow**comprVel), axis=-1)
            ulm_v_rgb = brighten_map(ulm_v_rgb, brighten)

        if show_matplotlib:
            fig, ax = plt.subplots(figsize=figsize)
            im = imagesc(
                x_axis,
                z_axis,
                ulm_v_rgb,
                cmap=cmapv,
            )

            if title is True:
                ax.set_title("ULM - Velocity Magnitude")
            else:
                ax.set_title(title)

            if axis:
                ax.set_xlabel("Width (mm)")
                ax.set_ylabel("Depth (mm)")

                if colorbar:
                    fig, cax = add_colorbar(fig, ax, im)
                    y_ticks = cax.get_yticks()
                    n_ticks = len(y_ticks)
                    labels = np.round(
                        [i * vmax / (n_ticks - 1) for i in range(n_ticks)]
                    ).astype(int)
                    cax.set_ylabel("Velocity (mm/s)")
                    cax.set_yticks(y_ticks)
                    cax.set_yticklabels(labels)
            else:
                ax.axis("off")

        image = matplotlib_to_pil(ulm_v_rgb)
        output_images["ulm_v"] = image
        if show_pillow:
            image.show()

        if save_dir:
            file_path = Path(save_dir, f"ulm_v.{filetype}")
            save_image(image, fig, file_path, tag=tag)

    if show_matplotlib:
        plt.show()

    return output_images


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process tracks from file path")
    parser.add_argument(
        "file_path",
        type=Path,
        help="Path to the file",  # default=".temp/tracks.hdf5"
    )
    # Post process tracking arguments
    parser.add_argument(
        "--upsampling_factor", type=int, default=8, help="Upsampling factor"
    )
    parser.add_argument(
        "--min_length", type=int, default=10, help="Minimum track length"
    )
    parser.add_argument(
        "--max_linking_distance",
        type=float,
        default=10,
        help="Maximum linking distance",
    )
    parser.add_argument(
        "--smooth_factor", type=float, default=None, help="Smooth factor, must be odd."
    )
    # Visualization arguments
    parser.add_argument(
        "--saturation", type=float, default=0.8, help="Saturation value"
    )
    parser.add_argument("--brighten", type=float, default=0.4, help="Brighten value")
    parser.add_argument(
        "--sigma_gauss", type=float, default=0.1, help="Sigma value for Gaussian filter"
    )
    parser.add_argument(
        "--comprVel", type=float, default=0.25, help="Compression value for velocity"
    )
    parser.add_argument(
        "--comprDen", type=float, default=0.33, help="Compression value for density"
    )
    parser.add_argument("--shadow", type=float, default=0.3, help="Shadow value")
    parser.add_argument(
        "--save_dir", type=str, default="./images", help="Save directory"
    )
    parser.add_argument(
        "--postprocessed",
        action="store_true",
        help="Tracks are already postprocessed, so skip postprocessing.",
    )
    parser.add_argument(
        "--filetype",
        type=str,
        default="png",
        choices=["pdf", "png", "jpg"],
        help="File type to save",
    )
    parser.add_argument(
        "--style", type=str, default="ius", choices=["ius", "pala"], help="Plot style"
    )
    parser.add_argument(
        "--show_matplotlib", action="store_true", help="Show matplotlib plot"
    )
    parser.add_argument("--show_pillow", action="store_true", help="Show PIL image")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Path to config file, will override all other arguments.",
        default=None,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    file_path = args.file_path

    tracks_obj = ReadTracks(file_path)
    tracks_obj.read()
    if args.postprocessed:
        tracks_obj.tracks_post = tracks_obj.tracks

    if args.config:
        config_path = args.config
        config = load_config_from_yaml(config_path)
        config = Config(config)

        print(f"Loaded config: {yellow(config_path)}")
        print("Using config params for visualization.")
        save_dir = args.save_dir
        args = config.visualization
        args.upsampling_factor = config.tracking.upscale
        args.min_length = config.tracking.min_length
        args.max_linking_distance = config.tracking.max_linking_distance
        args.smooth_factor = config.tracking.smooth_factor
        args.save_dir = save_dir

    (
        density_map,
        velocity_map_x,
        velocity_map_z,
        velocity_map_norm,
    ) = tracks_obj.tracks_to_maps(
        upsampling_factor=args.upsampling_factor,
        min_length=args.min_length,
        max_linking_distance=args.max_linking_distance,
        smooth_factor=args.smooth_factor,
    )

    show_ulm(
        ulm_den=density_map,
        ulm_v=velocity_map_norm,
        ulm_vx=velocity_map_x,
        ulm_vz=velocity_map_z,
        saturation=args.saturation,
        brighten=args.brighten,
        sigma_gauss=args.sigma_gauss,
        comprVel=args.comprVel,
        comprDen=args.comprDen,
        shadow=args.shadow,
        save_dir=args.save_dir,
        filetype=args.filetype,
        style=args.style,
        show_matplotlib=args.show_matplotlib,
    )
