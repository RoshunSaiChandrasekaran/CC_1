"""
ga.py
Core evolutionary algorithm for the Cookie/Cake Recipe Generator.
  - A multi-factor fitness function (role coverage, ratio quality,
    recipe-size sanity, mix-in variety) instead of a plain ingredient
    count
  - Role-aware mutation (ingredient substitution prefers same-role
    ingredients most of the time, to keep recipes coherent, but still
    allows fully random substitution some of the time for novelty)
  - Baking-appropriate normalisation (scaled relative to the flour /
    "structure" weight, rather than to a fixed total volume as in the
    original soup-based PIERRE)
  - Descriptive name generation based on the recipe's dominant
    mix-in/flavour ingredient
"""

import copy
import json
import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Constants describing what makes a "valid"/"good" baked good.
# These bands were set by eyeballing typical ratios in the inspiring set
# (see data/inspiring_set.json) and standard baking references; they are
# deliberately a rough, declared approximation rather than ground truth -
# see the report for a discussion of this choice and its limitations.
# ---------------------------------------------------------------------------

ESSENTIAL_ROLES = ["structure", "fat", "sweetener"]
BONUS_ROLES = ["binding", "leavening", "liquid", "flavour", "mix-in"]

# acceptable ratio of (role weight) / (structure weight)
RATIO_BANDS = {
    "fat": (0.25, 1.0),
    "sweetener": (0.25, 1.3),
}

MIN_INGREDIENTS = 4
MAX_INGREDIENTS = 14
TARGET_STRUCTURE_WEIGHT = 250  # grams of "structure" ingredients a normalised recipe should contain


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
# Fitness
# ---------------------------------------------------------------------------

def weight_by_role(recipe, role):
    return sum(i["amount"] for i in recipe["ingredients"] if i["role"] == role and i["unit"] in ("g", "ml"))


def evaluate_recipe(recipe):
    """Multi-factor fitness function.

    Replaces Simple PIERRE's "fitness = number of ingredients" with a
    score that rewards recipes that (a) actually contain the roles needed
    to function as a baked good, (b) have fat/sweetener in a sensible
    proportion to the structural ingredients, (c) are a sensible size,
    and (d) have some variety of mix-in/flavour ingredients (a proxy for
    interestingness/novelty).
    """
    roles_present = {i["role"] for i in recipe["ingredients"]}
    score = 0.0

    # --- Validity: must contain the essential roles, or be heavily penalised
    missing_essential = [r for r in ESSENTIAL_ROLES if r not in roles_present]
    if missing_essential:
        score -= 15 * len(missing_essential)

    # --- Bonus roles present (binding, leavening, liquid, flavour, mix-in)
    for r in BONUS_ROLES:
        if r in roles_present:
            score += 3

    # --- Ratio quality, relative to structure weight
    structure_weight = weight_by_role(recipe, "structure")
    if structure_weight > 0:
        for role, (low, high) in RATIO_BANDS.items():
            role_weight = weight_by_role(recipe, role)
            ratio = role_weight / structure_weight
            if low <= ratio <= high:
                score += 5
            else:
                # partial credit the closer it is to the band
                distance = min(abs(ratio - low), abs(ratio - high))
                score += max(0, 5 - distance * 5)
    else:
        score -= 10  # no structural ingredients at all -> not really a recipe

    # --- Recipe-size sanity (too few or too many ingredients is unrealistic)
    n = len(recipe["ingredients"])
    if n < MIN_INGREDIENTS:
        score -= (MIN_INGREDIENTS - n) * 4
    elif n > MAX_INGREDIENTS:
        score -= (n - MAX_INGREDIENTS) * 2

    # --- Mix-in / flavour variety bonus (proxy for interestingness), capped
    interesting_roles = {"mix-in", "flavour"}
    variety = len({i["ingredient"] for i in recipe["ingredients"] if i["role"] in interesting_roles})
    score += min(variety, 4) * 2

    return score


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

def mutate_recipe(recipe, all_ingredients, by_role, same_role_prob=0.7):
    op = random.randint(0, 3)
    ingredients = recipe["ingredients"]

    if op == 0 and ingredients:
        # change the amount of a random ingredient by +/- 10-30%
        i = random.randrange(len(ingredients))
        ingredients[i] = dict(ingredients[i])
        delta = random.uniform(-0.3, 0.3)
        ingredients[i]["amount"] = max(1, int(ingredients[i]["amount"] * (1 + delta)))

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
                i["amount"] = max(1, int(i["amount"] * scale))
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


def select_population(population, new_recipes):
    new_recipes.sort(key=lambda r: r["fitness"], reverse=True)
    half = len(population) // 2
    combined = population[:half] + new_recipes[:half]
    combined.sort(key=lambda r: r["fitness"], reverse=True)
    return combined


@dataclass
class RunResult:
    best_fitness_history: list = field(default_factory=list)
    mean_fitness_history: list = field(default_factory=list)
    final_population: list = field(default_factory=list)


def run_evolution(recipes, population_size=30, generations=100, seed=None):
    if seed is not None:
        random.seed(seed)

    all_ingredients = collect_all_ingredients(recipes)
    by_role = ingredients_by_role(all_ingredients)

    population = init_population(recipes, population_size)

    result = RunResult()
    for gen in range(generations):
        new_recipes = generate_recipes(population_size, population, all_ingredients, by_role)
        population = select_population(population, new_recipes)
        result.best_fitness_history.append(population[0]["fitness"])
        result.mean_fitness_history.append(sum(r["fitness"] for r in population) / len(population))

    for r in population:
        if r["name"].startswith("recipe "):
            name_recipe(r)

    result.final_population = population
    return result
