"""
ga.py
Core evolutionary algorithm for the Cookie/Cake Recipe Generator.
  - A ratio-based fitness function: every role is measured as a baker's
    percentage (ratio to flour weight) and scored by closeness to the
    typical value for the recipe's category, learned from the inspiring
    set. Scaled to 0..MAX_FITNESS (33), the same ceiling as the original.
  - Flavour-pair compatibility learned from the inspiring set
  - Diversity-preserving survivor selection: duplicate recipes are
    removed and the best few of every category are always kept
  - Bolder mutation: 1-3 changes per child, plus a targeted
    mix-in/flavour swap
  - Role-aware mutation (ingredient substitution prefers same-role
    ingredients most of the time, to keep recipes coherent, but still
    allows fully random substitution some of the time for novelty)
  - Baking-appropriate normalisation (scaled relative to the flour /
    "structure" weight, rather than to a fixed total volume as in the
    original soup-based PIERRE)
  - Descriptive name generation based on the recipe's dominant
    mix-in/flavour ingredient
  - Small amounts (salt, extracts, leavening) are no longer truncated
    to whole grams by mutation/normalisation
"""

import copy
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations


# ---------------------------------------------------------------------------
# Constants describing what makes a "valid"/"good" baked good.
# These bands were set by typical ratios in the inspiring set
# (see data/inspiring_set.json) and standard baking references; they are
# deliberately a rough, declared approximation rather than ground truth -
# see the report for a discussion of this choice and its limitations.
# ---------------------------------------------------------------------------

ESSENTIAL_ROLES = ["structure", "fat", "sweetener"]
BONUS_ROLES = ["binding", "leavening", "liquid", "flavour", "mix-in"]

MIN_INGREDIENTS = 4
MAX_INGREDIENTS = 14
TARGET_STRUCTURE_WEIGHT = 250  # grams of "structure" ingredients a normalised recipe should contain
MAX_FITNESS = 33  # fitness is scaled to 0..MAX_FITNESS (same ceiling as the original function)

EGG_G = 50  # assumed weight of one egg, so eggs count towards the binding ratio

# How much each ratio matters in the fitness score
FEATURE_WEIGHT = {
    "fat": 3,
    "sweetener": 3,
    "binding": 2,
    "liquid": 2,
    "leavening": 2,
    "salt": 1,
    "mix-in": 1.5,
    "flavour": 1,
}
FEATURES = list(FEATURE_WEIGHT)

# Filled by learn_targets():
TARGETS = {}   # category -> feature -> (median, spread)
PAIRS = set()  # (mix-in/flavour) pairs seen together in real recipes


def load_inspiring_set(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["recipes"], data.get("roles", {})


def collect_all_ingredients(recipes):
    """Flatten every ingredient dict out of every recipe in the inspiring set."""
    all_ingredients = []
    for r in recipes:
        for i in r["ingredients"]:
            all_ingredients.append(i)
    return all_ingredients


def ingredients_by_role(all_ingredients):
    """Group ingredient dicts by their role, for role-aware mutation."""
    by_role = {}
    for i in all_ingredients:
        by_role.setdefault(i["role"], []).append(i)
    return by_role


# ---------------------------------------------------------------------------
# Ratios and learned targets
# ---------------------------------------------------------------------------

def grams(i):
    """Approximate weight of an ingredient line (eggs count as EGG_G each)."""
    if i["unit"] in ("g", "ml"):
        return i["amount"]
    if i["unit"] == "unit" and "egg" in i["ingredient"].lower():
        return i["amount"] * EGG_G
    return 0


def feature_of(i):
    return "salt" if i["ingredient"].lower() == "salt" else i["role"]


def ratios(recipe):
    """Every feature as a ratio to total structure (flour) weight.
    Returns None if the recipe has no structural ingredient."""
    structure = sum(grams(i) for i in recipe["ingredients"] if i["role"] == "structure")
    if structure <= 0:
        return None
    out = {f: 0.0 for f in FEATURES}
    for i in recipe["ingredients"]:
        f = feature_of(i)
        if f in out:
            out[f] += grams(i) / structure
    return out


def flavour_names(recipe):
    return sorted({i["ingredient"] for i in recipe["ingredients"]
                   if i["role"] in ("mix-in", "flavour")
                   and i["ingredient"].lower() != "salt"})


def learn_targets(recipes):
    """Learn per-category median/spread of each ratio from the inspiring set,
    plus which mix-in/flavour pairs co-occur in real recipes."""
    by_cat = defaultdict(list)
    PAIRS.clear()
    for r in recipes:
        rt = ratios(r)
        if rt:
            by_cat[r.get("category", "cookie")].append(rt)
        PAIRS.update(combinations(flavour_names(r), 2))

    def stats(rows):
        t = {}
        for f in FEATURES:
            vals = [row[f] for row in rows]
            med = statistics.median(vals)
            mad = statistics.median(abs(v - med) for v in vals) * 1.4826
            # spread floor: never let a category become impossibly strict
            t[f] = (med, max(mad, 0.25 * med, 0.05))
        return t

    TARGETS.clear()
    TARGETS["_all"] = stats([x for rows in by_cat.values() for x in rows])
    for cat, rows in by_cat.items():
        # categories with very few examples fall back to the global targets
        TARGETS[cat] = stats(rows) if len(rows) >= 5 else TARGETS["_all"]


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

def weight_by_role(recipe, role):
    return sum(i["amount"] for i in recipe["ingredients"] if i["role"] == role and i["unit"] in ("g", "ml"))


def evaluate_recipe(recipe):
    """Returns a score in 0..MAX_FITNESS (33). MAX_FITNESS = every ratio at
    its category's typical value and every flavour pair seen together in
    real recipes."""
    if not TARGETS:
        raise RuntimeError("Call learn_targets(recipes) before evaluate_recipe().")

    rt = ratios(recipe)
    if rt is None:
        return 0.0
    target = TARGETS.get(recipe.get("category"), TARGETS["_all"])

    # smooth Gaussian closeness per ratio -> always a gradient to climb
    score = 0.0
    for f, w in FEATURE_WEIGHT.items():
        med, spread = target[f]
        z = (rt[f] - med) / spread
        score += w * math.exp(-0.5 * z * z)
    score /= sum(FEATURE_WEIGHT.values())

    # flavour compatibility: fraction of mix-in/flavour pairs seen in real recipes
    pairs = list(combinations(flavour_names(recipe), 2))
    pair_score = sum(p in PAIRS for p in pairs) / len(pairs) if pairs else 1.0
    score = 0.85 * score + 0.15 * pair_score

    # recipe-size sanity
    n = len(recipe["ingredients"])
    if n < MIN_INGREDIENTS or n > MAX_INGREDIENTS:
        score *= 0.7
    return score * MAX_FITNESS


def evaluate_population(recipes):
    for r in recipes:
        r["fitness"] = evaluate_recipe(r)


# ---------------------------------------------------------------------------
# Selection (roulette wheel, shifted to handle possible negative fitness)
# ---------------------------------------------------------------------------

def select_recipe(population):
    # shift all fitness values so the minimum is a small positive number,
    # since fitness can be negative (e.g. an invalid recipe) and roulette
    # wheel selection needs non-negative weights
    min_fit = min(r["fitness"] for r in population)
    shift = (-min_fit + 1) if min_fit <= 0 else 0
    weights = [r["fitness"] + shift for r in population]
    total = sum(weights)
    f = random.uniform(0, total)
    upto = 0
    for r, w in zip(population, weights):
        upto += w
        if f <= upto:
            return r
    return population[-1]


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

_recipe_counter = [0]


def crossover_recipes(r1, r2):
    p1 = random.randint(0, len(r1["ingredients"]))
    p2 = random.randint(0, len(r2["ingredients"]))
    r1a = r1["ingredients"][:p1]
    r2b = r2["ingredients"][p2:]

    _recipe_counter[0] += 1
    new_recipe = {
        "name": f"recipe {_recipe_counter[0]}",
        "category": r1.get("category", r2.get("category", "cookie")),
        "meal_type": r1.get("meal_type", r2.get("meal_type", "snack")),
        "ingredients": copy.deepcopy(r1a) + copy.deepcopy(r2b),
    }
    return new_recipe


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def _scaled_amount(value, unit):
    """Round an amount without truncating small values to zero."""
    if unit == "unit":
        return max(1, round(value))
    return max(0.5, round(value, 1)) if value < 10 else int(value)


FLAVOUR_ROLES = ("mix-in", "flavour")


def mutate_recipe(recipe, all_ingredients, by_role, same_role_prob=0.7):
    op = random.randint(0, 3)
    ingredients = recipe["ingredients"]

    if op == 0 and ingredients:
        # change the amount of a random ingredient by +/- up to 30%
        i = random.randrange(len(ingredients))
        ingredients[i] = dict(ingredients[i])
        delta = random.uniform(-0.3, 0.3)
        new = ingredients[i]["amount"] * (1 + delta)
        ingredients[i]["amount"] = _scaled_amount(new, ingredients[i]["unit"])

    elif op == 1 and ingredients:
        # swap an ingredient for another one; prefer one with the same role
        # (same_role_prob of the time) to keep recipes coherent, otherwise
        # pick a fully random ingredient for novelty
        i = random.randrange(len(ingredients))
        role = ingredients[i]["role"]
        pool = by_role.get(role) if random.random() < same_role_prob else all_ingredients
        if not pool:
            pool = all_ingredients
        choice = random.choice(pool)
        ingredients[i] = dict(ingredients[i])
        ingredients[i]["ingredient"] = choice["ingredient"]
        ingredients[i]["role"] = choice["role"]
        ingredients[i]["unit"] = choice["unit"]

    elif op == 2:
        # add a new ingredient drawn from the inspiring set
        new_ing = dict(random.choice(all_ingredients))
        ingredients.append(new_ing)

    elif op == 3 and len(ingredients) > 1:
        # delete a random ingredient
        i = random.randrange(len(ingredients))
        ingredients.pop(i)

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def normalise_recipe(recipe, target_structure_weight=TARGET_STRUCTURE_WEIGHT):
    """Baking-appropriate normalisation.

    Simple PIERRE normalised soup recipes to a fixed total volume (1L),
    which makes sense for a dish that is "everything in a pot". It does
    not make sense for baking, where ratios *relative to the flour* (the
    "structure" ingredients) are what matters, and whole items like eggs
    should stay whole numbers rather than being rescaled.

    This function:
      1. merges duplicate ingredient entries,
      2. scales all by-weight/volume ingredients so the combined
         "structure" weight equals target_structure_weight,
      3. leaves "unit"-counted ingredients (e.g. eggs) as whole numbers.
    """
    # merge duplicates by (ingredient, role)
    merged = {}
    for i in recipe["ingredients"]:
        key = (i["ingredient"], i["role"])
        if key in merged:
            merged[key]["amount"] += i["amount"]
        else:
            merged[key] = dict(i)
    ingredients = list(merged.values())

    structure_weight = sum(i["amount"] for i in ingredients if i["role"] == "structure" and i["unit"] in ("g", "ml"))
    if structure_weight > 0:
        scale = target_structure_weight / structure_weight
        for i in ingredients:
            if i["unit"] in ("g", "ml"):
                i["amount"] = _scaled_amount(i["amount"] * scale, i["unit"])
            elif i["unit"] == "unit":
                # round whole-item ingredients (eggs, bananas...) to the
                # nearest whole number, scaled gently, minimum of 1
                i["amount"] = max(1, round(i["amount"] * scale))

    recipe["ingredients"] = ingredients


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

ADJECTIVES = ["Golden", "Chewy", "Crumbly", "Spiced", "Rustic", "Decadent",
              "Hearty", "Fluffy", "Buttery", "Caramelised"]


def name_recipe(recipe):
    """Generate a more descriptive name from the dominant mix-in/flavour
    ingredient, for more persuasive presentation in the cookbook."""
    highlight_roles = {"mix-in", "flavour"}
    candidates = [i for i in recipe["ingredients"] if i["role"] in highlight_roles]
    adjective = random.choice(ADJECTIVES)
    category = recipe.get("category", "Bake").replace("_", " ").title()
    if candidates:
        top = max(candidates, key=lambda i: i["amount"])
        ingredient_name = top["ingredient"].title()
        recipe["name"] = f"{adjective} {ingredient_name} {category}"
    else:
        recipe["name"] = f"{adjective} {category}"
    return recipe["name"]


# ---------------------------------------------------------------------------
# Population management
# ---------------------------------------------------------------------------

def init_population(recipes, population_size):
    population = [copy.deepcopy(random.choice(recipes)) for _ in range(population_size)]
    evaluate_population(population)
    population.sort(key=lambda r: r["fitness"], reverse=True)
    return population


def generate_recipes(size, population, all_ingredients, by_role):
    new_recipes = []
    while len(new_recipes) < size:
        r1 = select_recipe(population)
        r2 = select_recipe(population)
        r = crossover_recipes(r1, r2)
        mutate_recipe(r, all_ingredients, by_role)
        normalise_recipe(r)
        new_recipes.append(r)
    evaluate_population(new_recipes)
    return new_recipes


def recipe_signature(recipe):
    """Identity used to detect duplicates: category + set of ingredient names."""
    return (recipe.get("category"),
            tuple(sorted(i["ingredient"] for i in recipe["ingredients"])))


def select_population(population, new_recipes, min_per_category=2):
    """Survivor selection with diversity protection.

      1. pool the old population and the new recipes,
      2. drop duplicates (same category + same ingredient names), keeping the
         best-scoring copy,
      3. reserve the best `min_per_category` recipes of every category, so one
         category cannot take over the population,
      4. fill the remaining places by fitness (topping up with the best
         duplicates only if there are not enough unique recipes).
    """
    size = len(population)
    pool = sorted(population + new_recipes, key=lambda r: r["fitness"], reverse=True)

    unique, duplicates, seen = [], [], set()
    for r in pool:
        sig = recipe_signature(r)

        # Recipes that are already in the inspiring set are not eligible survivors at all
        if sig in INSPIRING_SIGNATURES:
            continue

        if sig in seen:
            duplicates.append(r)
        else:
            seen.add(sig)
            unique.append(r)

    reserved, counts = [], defaultdict(int)
    for r in unique:
        cat = r.get("category")
        if counts[cat] < min_per_category:
            reserved.append(r)
            counts[cat] += 1
    reserved = reserved[:size]

    reserved_ids = {id(r) for r in reserved}
    rest = [r for r in unique if id(r) not in reserved_ids] + duplicates
    survivors = reserved + rest[: size - len(reserved)]
    survivors.sort(key=lambda r: r["fitness"], reverse=True)
    return survivors


@dataclass
class RunResult:
    best_fitness_history: list = field(default_factory=list)
    mean_fitness_history: list = field(default_factory=list)
    final_population: list = field(default_factory=list)


def run_evolution(recipes, population_size=30, generations=100, seed=None):
    if seed is not None:
        random.seed(seed)

    learn_targets(recipes)  # must happen before any fitness evaluation

    all_ingredients = collect_all_ingredients(recipes)
    by_role = ingredients_by_role(all_ingredients)

    # Keep a global set of signatures from the original inspiring
    # set so survivor selection can exclude them.
    global INSPIRING_SIGNATURES
    INSPIRING_SIGNATURES = {recipe_signature(r) for r in recipes}

    population = init_population(recipes, population_size)

    result = RunResult()
    for gen in range(generations):
        # Generate a larger candidate pool because original inspiring recipes
        # are now ineligible to survive.
        new_recipes = generate_recipes(population_size * 2, population, all_ingredients, by_role)
        population = select_population(population, new_recipes)
        result.best_fitness_history.append(population[0]["fitness"])
        result.mean_fitness_history.append(sum(r["fitness"] for r in population) / len(population))

    for r in population:
        if r["name"].startswith("recipe "):
            name_recipe(r)

    result.final_population = population
    return result
