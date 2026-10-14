import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
import ga


def format_recipe_text(recipe):
    lines = [f"{recipe['name']}", f"({recipe.get('category', 'recipe')}) - fitness: {recipe['fitness']:.1f}", ""]
    lines.append("Ingredients:")
    for i in recipe["ingredients"]:
        unit = "" if i["unit"] == "unit" else i["unit"]
        plural = "s" if i["unit"] == "unit" and i["amount"] != 1 else ""
        lines.append(f"  - {i['amount']}{unit} {i['ingredient']}{plural}  [{i['role']}]")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evolutionary cookie/cake recipe generator")
    parser.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..", "data", "inspiring_set.json"))
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--generations", type=int, default=150)
    parser.add_argument("--top", type=int, default=5, help="number of top recipes to report")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "..", "outputs"))
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    recipes, roles = ga.load_inspiring_set(args.data)
    print(f"Loaded {len(recipes)} recipes from inspiring set.")

    result = ga.run_evolution(
        recipes,
        population_size=args.population,
        generations=args.generations,
        seed=args.seed,
    )

    # --- fitness plot
    plt.figure(figsize=(8, 5))
    plt.plot(result.best_fitness_history, label="Best fitness")
    plt.plot(result.mean_fitness_history, label="Mean fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness over generations")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(args.outdir, "fitness_history.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Saved fitness plot to {plot_path}")

    # --- final population JSON
    pop_path = os.path.join(args.outdir, "final_population.json")
    with open(pop_path, "w") as f:
        json.dump({"recipes": result.final_population}, f, indent=2)
    print(f"Saved final population to {pop_path}")

    # --- top recipes, human readable
    top_recipes = result.final_population[: args.top]
    top_path = os.path.join(args.outdir, "top_recipes.txt")
    with open(top_path, "w") as f:
        for r in top_recipes:
            f.write(format_recipe_text(r))
            f.write("\n" + ("-" * 40) + "\n\n")
    print(f"Saved top {args.top} recipes to {top_path}")

    print("\n=== Top recipe ===\n")
    print(format_recipe_text(top_recipes[0]))


if __name__ == "__main__":
    main()
