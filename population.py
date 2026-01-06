import creature
import numpy as np


class Population:
    def __init__(self, pop_size, gene_count):
        self.creatures = [creature.Creature(gene_count=gene_count) for _ in range(pop_size)]

    @staticmethod
    def get_fitness_map(fits):
        fitmap = []
        total = 0.0
        for f in fits:
            total += float(f)
            fitmap.append(total)
        return fitmap

    @staticmethod
    def select_parent(fitmap):
        total = float(fitmap[-1])
        if total <= 0.0:
            # fallback: uniform random parent
            return int(np.random.randint(0, len(fitmap)))
        r = np.random.rand() * total
        for i, v in enumerate(fitmap):
            if r <= v:
                return i
        return len(fitmap) - 1
