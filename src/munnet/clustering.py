"""Этап cluster: кластеризация узлов разными методами.

По атрибутам (K-means, Ward, GMM, HDBSCAN), по сети (Louvain, Leiden, спектральная)
и на атрибутированной сети.
"""

from munnet.config import Config


def run(cfg: Config) -> None:
    raise NotImplementedError("ещё не реализован, см. PLAN.md, этап 3")
