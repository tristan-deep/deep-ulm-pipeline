# A tracking and visualization pipeline for ULM

The official implementation for the paper: [A Hybrid Deep Learning Pipeline for Improved Ultrasound Localization Microscopy](https://ieeexplore.ieee.org/document/9958562). Also available open access [here](https://tristan-deep.github.io/publications/ulm/ulm_paper/).

<p align="center">
    <img src="assets/ratbrain.png" alt="Rat Brain Visualization" width="600"/>
</p>

> [!TIP]
> **🚨 New:** localization model added (deepULM)! See this [script](./model.py).

## Overview
Performs the following steps:
- Load localization data from hdf5 file
- Load config file, see example in [tracking_config.yaml](./configs/tracking_config.yaml)
- Track localizations using track parameters from config file
- Save raw tracks to file
- Postprocess tracks
- Convert tracks to maps with [tracks_to_map.py](./tracks_to_map.py)
- Visualize and save maps using [visualization.py](./visualization.py)

### Example usage

```bash
python main.py -c ./configs/tracking_config.yaml
```

```bash
Args:
    -c (--config): path to tracking config file.
    -s (--save): Save raw tracks to file
    --skip: Skip folders that already have tracks.

```

> [!NOTE]
> Make sure the localizations are stored in the correct format (hdf5) and have only a single localization file per folder. You can refer to the [`read.py`](./read.py) file and the `ReadLocs` class for more information on the correct format.

### Save localizations
An example of how to save localizations to hdf5 format is shown in [save_localizations.py](./save_localizations.py). This script saves some dummy localizations to a file in the correct format. You should be able to run [main.py](./main.py) with the output of this script.

### Setup environment
Create a conda environment and pip install requirements.

```bash
conda create -n deep-ulm python=3.8 -y
conda activate deep-ulm
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
pip install -r requirements.txt
```

### Citation

If you use this pipeline in your research, please cite the following paper:

```
@inproceedings{stevens2022hybrid,
    title={A Hybrid Deep Learning Pipeline for Improved Ultrasound Localization Microscopy},
    author={Stevens, Tristan SW and Herbst, Elizabeth B and Luijten, Ben and Ossenkoppele, Boudewine W and Voskuil, Thierry J and Wang, Shiying and Youn, Jihwan and Errico, Claudia and Mischi, Massimo and Pezzotti, Nicola and others},
    booktitle={2022 IEEE International Ultrasonics Symposium (IUS)},
    pages={1--4},
    year={2022},
    organization={IEEE}
}
```