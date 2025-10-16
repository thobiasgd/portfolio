import random
import numpy as np
from multiprocessing import Pool
from aircraft_model import Aircraft
from evaluator import evaluate
from utils import plot_progress

class GeneticOptimizer:
    def __init__(self, pop_size, mutation_rate, generations, logger):
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.logger = logger
        self.population = []
        self.history = []

    # ----------------------------------------------
    def initialize_population(self):
        for i in range(self.pop_size):
            genes = [
                random.uniform(1.5, 2.5),   # envergadura
                random.uniform(0.3, 0.6),   # corda root
                random.uniform(0.2, 0.5),   # corda tip
                random.uniform(0, 20),      # sweep1
                random.uniform(0, 10),      # sweep2
                random.uniform(0, 6),       # incidencia
            ]
            self.population.append(Aircraft(genes, generation=0, index=i))

    # ----------------------------------------------
    def mutate(self, aircraft):
        for i in range(len(aircraft.genes)):
            if random.random() < self.mutation_rate:
                scale = 0.1 * aircraft.genes[i]
                aircraft.genes[i] += np.random.normal(0, scale)
        return aircraft

    # ----------------------------------------------
    def crossover(self, p1, p2, gen_idx, idx):
        cut = random.randint(1, len(p1.genes)-2)
        child_genes = p1.genes[:cut] + p2.genes[cut:]
        return Aircraft(child_genes, generation=gen_idx, index=idx)

    # ----------------------------------------------
    def evolve(self):
        self.initialize_population()

        for gen in range(self.generations):
            self.logger.info(f"🏁 Geração {gen+1}/{self.generations}")

            # modelagem + simulação paralela
            for a in self.population:
                a.modelar()
            with Pool() as pool:
                pool.map(lambda ac: (ac.simular(), evaluate(ac)), self.population)

            # ordena por fitness
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            best = self.population[0]
            self.history.append(best.fitness)
            self.logger.info(f"Melhor nota: {best.fitness:.6f}")

            # elitismo
            next_gen = [best]

            # reprodução
            while len(next_gen) < self.pop_size:
                p1, p2 = random.sample(self.population[:10], 2)
                child = self.crossover(p1, p2, gen+1, len(next_gen))
                child = self.mutate(child)
                next_gen.append(child)

            self.population = next_gen

        plot_progress(self.history, "evolucao.png")
        self.logger.info("Evolução concluída.")
        return self.population[0]
