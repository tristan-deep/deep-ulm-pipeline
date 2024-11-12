"""Load deepULM model
Author(s): Tristan Stevens
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


import keras
from huggingface_hub import HfApi, login, snapshot_download

from config import load_config_from_yaml
from models.classifier import classifier
from models.deepULM import deepULM
from utils import generate_fake_bmode_data, yellow


def load_model_from_hf(repo_id, revision="main"):
    """
    Load the model from a given repo_id using the Hugging Face library.
    Args:
        repo_id (str): The ID of the repository.
    Returns:
        The loaded model.

    """
    login(new_session=False)

    model_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
    )
    model, config = build_model(model_dir)
    api = HfApi()
    commit = api.list_repo_commits(repo_id, revision=revision)[0]
    commit_message = commit.title
    commit_time = commit.created_at.strftime("%B %d, %Y at %I:%M %p %Z")
    print(
        yellow(
            f"Succesfully loaded model {commit_message} from "
            f"'https://huggingface.co/{repo_id}'. Last updated on {commit_time}."
        )
    )

    return model, config


def build_model(model_path):
    """Build model from config found in model_path.
    Path to folder should contain a `config.yaml` and `weights.h5`.
    """

    model_path = Path(model_path)
    config = load_config_from_yaml(model_path / "config.yaml")

    if config.model.type == "deepULM":

        loc_model = deepULM(
            input_shape=(config.input_dim[0], config.input_dim[1], 1),
            fdim=4,
            N_maxpool=4,
            k_enc=config.model.kernel_size_enc,
            k_dec=config.model.kernel_size_dec,
            dropout_rate=(
                config.model.dropout_rate if "dropout_rate" in config.model else 0.5
            ),
            skip=config.model.skip,
            skip_sum=config.model.skip_sum if "skip_sum" in config.model else False,
            depth=config.model.depth,
            dilation=config.model.dilation,
            upscale=config.upscale,
            activation=config.model.activation,
        )

    else:
        raise NotImplementedError(f"Model type {config.model.type} not implemented.")

    # Full model
    inputs, outputs = {}, {}
    inputs["input"] = keras.layers.Input(shape=loc_model.input.shape[1:])

    loc_output = loc_model(inputs["input"])

    outputs["localizations"] = loc_output

    if config.classifier:
        clas_model = classifier(
            loc_model.output.shape[1:],
            config.classifier.kmax,
            config.classifier.kmin,
            config.classifier.depth,
        )

        outputs["binary"] = clas_model(loc_output)

    model = keras.models.Model(inputs=inputs, outputs=outputs)

    ckpt_file = model_path / "weights.h5"
    model.load_weights(ckpt_file)
    return model, config


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt

    from utils import find_peaks_tensorflow

    ## load custom model from path
    # model_path = "path_to_model_folder/
    # model, config = load_model(model_path)
    # or
    ## load model from huggingface
    model, config = load_model_from_hf("tristan-deep/deepULM", "main")

    print(model.summary())

    # Example usage
    fake_bmode_data = generate_fake_bmode_data(
        config.input_dim, num_blobs=10, std_dev_range=(1, 3)
    )

    output = model.predict(fake_bmode_data[None, ..., None])
    peaks = find_peaks_tensorflow(output["binary"], 0.3)[0]

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(fake_bmode_data, cmap="gray")
    axs[1].imshow(output["localizations"][0, ..., 0], cmap="viridis")
    axs[2].imshow(output["binary"][0, ..., 0], cmap="gray")
    axs[2].scatter(
        peaks[:, 2],
        peaks[:, 1],
        marker="x",
        color="r",
        s=3,
    )

    path = "test_localization_model.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved image to {path}")
