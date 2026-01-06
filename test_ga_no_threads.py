import argparse
import csv
import random
import shutil
from pathlib import Path

import numpy as np

import creature
import genome
import population
import simulation


def run_ga(
    run_name: str,
    pop_size: int,
    init_gene_count: int,
    generations: int,
    iterations_per_creature: int,
    point_mutate_rate: float,
    point_mutate_amount: float,
    shrink_rate: float,
    grow_rate: float,
    seed: int,
    clean: bool,
):
    # Reproducibility (as much as possible)
    np.random.seed(seed)
    random.seed(seed)

    run_dir = Path("runs") / run_name
    elites_dir = run_dir / "elites"
    metrics_path = run_dir / "metrics.csv"

    if clean and run_dir.exists():
        shutil.rmtree(run_dir)

    elites_dir.mkdir(parents=True, exist_ok=True)

    pop = population.Population(pop_size=pop_size, gene_count=init_gene_count)
    sim = simulation.Simulation()

    best_so_far = -1e9
    best_gen = -1

    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gen", "best_fitness", "mean_fitness", "std_fitness", "mean_links", "max_links"])

        for gen in range(generations):
            # Evaluate
            for cr in pop.creatures:
                sim.run_creature(cr, iterations_per_creature)

            fits = [cr.get_fitness() for cr in pop.creatures]
            links = [len(cr.get_expanded_links()) for cr in pop.creatures]

            best_fit = float(np.max(fits))
            mean_fit = float(np.mean(fits))
            std_fit = float(np.std(fits))
            mean_links = float(np.mean(links))
            max_links = int(np.max(links))

            writer.writerow([gen, best_fit, mean_fit, std_fit, mean_links, max_links])
            f.flush()

            print(
                gen,
                "best:", np.round(best_fit, 4),
                "mean:", np.round(mean_fit, 4),
                "std:", np.round(std_fit, 4),
                "mean_links:", int(np.round(mean_links)),
                "max_links:", max_links,
            )

            # Save elite ONLY when we improved (so you don't get 300 files every run)
            if best_fit > best_so_far + 1e-9:
                best_so_far = best_fit
                best_gen = gen
                elite_src = pop.creatures[int(np.argmax(fits))]

                # overwrite “best so far”
                genome.Genome.to_csv(elite_src.dna, str(run_dir / "best_elite.csv"))

                # also keep a checkpoint of “new best moments”
                genome.Genome.to_csv(elite_src.dna, str(elites_dir / f"elite_gen_{gen:04d}.csv"))

            # Selection -> Next generation
            fit_map = population.Population.get_fitness_map(fits)
            new_creatures = []

            for _ in range(pop_size):
                p1 = pop.creatures[population.Population.select_parent(fit_map)]
                p2 = pop.creatures[population.Population.select_parent(fit_map)]

                dna = genome.Genome.crossover(p1.dna, p2.dna)
                dna = genome.Genome.point_mutate(dna, rate=point_mutate_rate, amount=point_mutate_amount)
                dna = genome.Genome.shrink_mutate(dna, rate=shrink_rate)
                dna = genome.Genome.grow_mutate(dna, rate=grow_rate)

                child = creature.Creature(1)
                child.update_dna(dna)
                new_creatures.append(child)

            # Elitism: copy the best from this generation to slot 0
            elite_src = pop.creatures[int(np.argmax(fits))]
            elite = creature.Creature(1)
            elite.update_dna(elite_src.dna)
            new_creatures[0] = elite

            pop.creatures = new_creatures

    print("\n=== RUN COMPLETE ===")
    print("Run:", run_name)
    print("Best fitness:", best_so_far)
    print("Best generation:", best_gen)
    print("Saved:", metrics_path)
    print("Saved:", run_dir / "best_elite.csv")
    print("Elite checkpoints in:", elites_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="run_01")
    parser.add_argument("--pop", type=int, default=10)
    parser.add_argument("--genes", type=int, default=3)
    parser.add_argument("--gens", type=int, default=300)
    parser.add_argument("--iters", type=int, default=2400)

    parser.add_argument("--mut-rate", type=float, default=0.10)
    parser.add_argument("--mut-amt", type=float, default=0.25)
    parser.add_argument("--shrink", type=float, default=0.25)
    parser.add_argument("--grow", type=float, default=0.10)

    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--clean", action="store_true")

    args = parser.parse_args()

    run_ga(
        run_name=args.run,
        pop_size=args.pop,
        init_gene_count=args.genes,
        generations=args.gens,
        iterations_per_creature=args.iters,
        point_mutate_rate=args.mut_rate,
        point_mutate_amount=args.mut_amt,
        shrink_rate=args.shrink,
        grow_rate=args.grow,
        seed=args.seed,
        clean=args.clean,
    )
