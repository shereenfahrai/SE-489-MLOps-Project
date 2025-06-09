import cProfile
import pstats
from pstats import SortKey

from omegaconf import OmegaConf

# ruff: noqa: F401
from fake_news_detection.train_model import train_with_cfg

if __name__ == "__main__":
    cfg = OmegaConf.load("../config/config.yaml")

    cProfile.runctx("train_with_cfg(cfg)", globals(), locals(), "reports/figures/train_profile.prof")

    stats = pstats.Stats("reports/figures/train_profile.prof")
    stats.sort_stats(SortKey.CUMULATIVE).print_stats(30)
