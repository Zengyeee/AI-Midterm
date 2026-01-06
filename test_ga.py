# If you on a Windows machine with any Python version
# or an M1 mac with any Python version
# or an Intel Mac with Python > 3.7
# this multi-threaded version does not work
# please use test_ga_no_threads.py on those setups

import unittest
import population
import simulation
import genome
import creature
import numpy as np


class TestGA(unittest.TestCase):
    def testBasicGA(self):
        POP_SIZE = 10
        INIT_GENE_COUNT = 3
        ITERATIONS_PER_CREATURE = 2400
        GENERATIONS = 300

        pop = population.Population(pop_size=POP_SIZE, gene_count=INIT_GENE_COUNT)

        # ThreadedSim uses multiprocessing. Might break on some Windows setups.
        sim = simulation.ThreadedSim(pool_size=1)
        # sim = simulation.Simulation()

        last_fits = None

        for gen in range(GENERATIONS):
            sim.eval_population(pop, ITERATIONS_PER_CREATURE)

            # NEW: mountain fitness
            fits = [cr.get_fitness() for cr in pop.creatures]
            last_fits = fits

            links = [len(cr.get_expanded_links()) for cr in pop.creatures]

            print(
                gen,
                "best:", np.round(np.max(fits), 3),
                "mean:", np.round(np.mean(fits), 3),
                "mean links:", int(np.round(np.mean(links))),
                "max links:", int(np.round(np.max(links))),
            )

            fit_map = population.Population.get_fitness_map(fits)

            new_creatures = []
            for _ in range(len(pop.creatures)):
                p1_ind = population.Population.select_parent(fit_map)
                p2_ind = population.Population.select_parent(fit_map)
                p1 = pop.creatures[p1_ind]
                p2 = pop.creatures[p2_ind]

                dna = genome.Genome.crossover(p1.dna, p2.dna)
                dna = genome.Genome.point_mutate(dna, rate=0.1, amount=0.25)
                dna = genome.Genome.shrink_mutate(dna, rate=0.25)
                dna = genome.Genome.grow_mutate(dna, rate=0.1)

                child = creature.Creature(1)
                child.update_dna(dna)
                new_creatures.append(child)

            # Elitism
            best_ind = int(np.argmax(fits))
            elite_src = pop.creatures[best_ind]
            elite = creature.Creature(1)
            elite.update_dna(elite_src.dna)
            new_creatures[0] = elite

            filename = f"elite_{gen}.csv"
            genome.Genome.to_csv(elite_src.dna, filename)

            pop.creatures = new_creatures

        self.assertIsNotNone(last_fits)
        self.assertGreaterEqual(float(np.max(last_fits)), 0.0)


unittest.main()
