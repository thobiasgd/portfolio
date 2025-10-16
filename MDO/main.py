from genetic_core import GeneticOptimizer
from utils import setup_logger

def main():
    logger = setup_logger()
    optimizer = GeneticOptimizer(
        pop_size=10,
        mutation_rate=0.1,
        generations=10,
        logger=logger
    )
    best = optimizer.evolve()
    logger.info(f"\nMelhor aeronave encontrada:\nGenes: {best.genes}\nFitness: {best.fitness:.6f}")

if __name__ == "__main__":
    main()
