import os
from pathlib import Path
import logging
import matplotlib.pyplot as plt

# ---------------- LOGGING ----------------
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger("GeneticVSP")

# ---------------- PATHS ----------------
def get_base_dir():
    base = Path(__file__).resolve().parent
    results = base / "resultados"
    results.mkdir(exist_ok=True)
    return results

# ---------------- PLOTS ----------------
def plot_progress(history, save_path):
    plt.figure(figsize=(8,5))
    plt.plot(history, marker="o")
    plt.title("Evolução da Aptidão")
    plt.xlabel("Geração")
    plt.ylabel("Nota de Avaliação Normalizada")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
