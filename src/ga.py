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
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations


# ---------------------------------------------------------------------------
# Constants describing what makes a "valid"/"good" baked good.
# Ideal ratios are now learned from the inspiring set (see learn_targets);
# the weights below are a declared design choice, not ground truth
# ---------------------------------------------------------------------------

ESSENTIAL_ROLES = ["structure", "fat", "sweetener"]
MIN_INGREDIENTS = 4
MAX_INGREDIENTS = 14
TARGET_STRUCTURE_WEIGHT = 250  # grams of "structure" ingredients a normalised recipe should contain
MAX_UNIT_COUNT = 6  # cap for whole-item ingredients (no recipe gets 10 eggs)
MIN_SHARE = 0.02  # g/ml ingredients below this share of the structure weight are dropped as crumbs...
TRACE_ROLES = ("seasoning", "leavening", "flavour")  # ...except roles that are small by nature
MAX_FITNESS = 33  # fitness is scaled to 0..MAX_FITNESS (same ceiling as before)
EGG_G = 50  # assumed weight of one egg, so eggs count towards the binding ratio
MISSING_ROLE_FACTOR = 0.3  # quality is multiplied by this for each missing essential role
SIZE_FACTOR = 0.7  # quality multiplier for too few / too many ingredients

# How much each ratio (relative to flour weight) matters in the quality score
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

TARGETS = {}  # category -> feature -> (median, spread); filled by learn_targets()
NOVELTY_WEIGHT = 0.2  # fitness = quality * (BASE + NOVELTY_WEIGHT*novelty + SURPRISE_WEIGHT*surprise)
SURPRISE_WEIGHT = 0.2  # bonus for one flavour/mix-in never seen in the recipe's category
BASE_WEIGHT = 1 - NOVELTY_WEIGHT - SURPRISE_WEIGHT
INSPIRING_RECIPES = []  # reference set for novelty; filled by run_evolution()
INSPIRING_SIGNATURES = set()  # seed signatures that may not survive; filled by run_evolution()
PAIRS = set()  # (mix-in/flavour) pairs seen together in real recipes; filled by learn_targets()
MEAL_FLAVOURS = defaultdict(set)  # meal_type -> mix-ins/flavours seen in real recipes of that meal type
CATEGORY_FLAVOURS = defaultdict(set)  # category -> mix-ins/flavours seen in real recipes of that category
MIXINS_BY_CATEGORY = defaultdict(list)  # category -> mix-in ingredient dicts, for the transplant mutation
QUALITY_W, PAIR_W, MEAL_W = 0.75, 0.15, 0.10  # blend inside the 0..1 quality score
MIN_DISTANCE = 0.15  # survivors must differ from each other by at least this Jaccard distance
MAX_CATEGORY_SHARE = 0.4  # no category may take more than this share of the population


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
    plus which flavour pairs co-occur and which flavours go with which meal type."""
    by_cat = defaultdict(list)
    PAIRS.clear()
    MEAL_FLAVOURS.clear()
    CATEGORY_FLAVOURS.clear()
    MIXINS_BY_CATEGORY.clear()
    for r in recipes:
        rt = ratios(r)
        if rt:
            by_cat[r.get("category", "cookie")].append(rt)
        PAIRS.update(combinations(flavour_names(r), 2))
        MEAL_FLAVOURS[r.get("meal_type")].update(flavour_names(r))
        CATEGORY_FLAVOURS[r.get("category")].update(flavour_names(r))
        for i in r["ingredients"]:
            if i["role"] == "mix-in":
                MIXINS_BY_CATEGORY[r.get("category")].append(dict(i))

    def stats(rows):
        t = {}
        for f in FEATURES:
            vals = [row[f] for row in rows]
            med = statistics.median(vals)
            mad = statistics.median(abs(v - med) for v in vals) * 1.4826
            # spread floor: never let a category become impossibly strict
            t[f] = (med, max(mad, 0.25 * med, 0.01))
        return t

    TARGETS.clear()
    TARGETS["_all"] = stats([x for rows in by_cat.values() for x in rows])
    for cat, rows in by_cat.items():
        # categories with very few examples fall back to the global targets
        TARGETS[cat] = stats(rows) if len(rows) >= 5 else TARGETS["_all"]


def gaussian_quality(recipe):
    """0..1: how close every ratio is to the typical value for the recipe's
    category. Smooth, so there is always a gradient to climb."""
    rt = ratios(recipe)
    if rt is None:
        return 0.0
    target = TARGETS.get(recipe.get("category"), TARGETS["_all"])
    score = 0.0
    for f, w in FEATURE_WEIGHT.items():
        med, spread = target[f]
        z = (rt[f] - med) / spread
        score += w * math.exp(-0.5 * z * z)
    return score / sum(FEATURE_WEIGHT.values())


def pair_score(recipe):
    """Fraction of mix-in/flavour pairs that appear together in real recipes."""
    pairs = list(combinations(flavour_names(recipe), 2))
    return sum(p in PAIRS for p in pairs) / len(pairs) if pairs else 1.0


def meal_score(recipe):
    """Fraction of mix-in/flavour ingredients seen in real recipes of the same meal type."""
    names = flavour_names(recipe)
    seen = MEAL_FLAVOURS.get(recipe.get("meal_type"), set())
    return sum(n in seen for n in names) / len(names) if names else 1.0


def evaluate_recipe(recipe):
    """Quality in 0..1 = ratio closeness, with role coverage and recipe size
    as multiplicative gates (no -15 penalties, so no cliffs)."""
    if not TARGETS:
        raise RuntimeError("Call learn_targets(recipes) before evaluate_recipe().")
    quality = (QUALITY_W * gaussian_quality(recipe)
               + PAIR_W * pair_score(recipe)
               + MEAL_W * meal_score(recipe))

    roles_present = {i["role"] for i in recipe["ingredients"]}
    missing = [r for r in ESSENTIAL_ROLES if r not in roles_present]
    quality *= MISSING_ROLE_FACTOR ** len(missing)

    n = len(recipe["ingredients"])
    if n < MIN_INGREDIENTS or n > MAX_INGREDIENTS:
        quality *= SIZE_FACTOR
    return quality


def recipe_ingredient_set(recipe):
    """Ingredient names used by the recipe, ignoring amounts and decorations
    (decorations have near-unique names, so counting them rewards padding)."""
    return {i["ingredient"] for i in recipe["ingredients"] if i["role"] != "decoration"}


def recipe_distance(r1, r2):
    """Jaccard distance between the ingredient-name sets of two recipes."""
    s1 = recipe_ingredient_set(r1)
    s2 = recipe_ingredient_set(r2)
    if not s1 and not s2:
        return 0.0
    return 1.0 - len(s1 & s2) / len(s1 | s2)


def novelty_score(recipe):
    """Novelty = distance to the most similar recipe in the inspiring set."""
    if not INSPIRING_RECIPES:
        return 0.0
    return min(recipe_distance(recipe, r) for r in INSPIRING_RECIPES)


def surprise_score(recipe):
    """1 if the recipe has at least one mix-in/flavour never seen in its own
    category, else 0. Capped at 1: one deliberate surprise counts, stacking
    several does not."""
    seen = CATEGORY_FLAVOURS.get(recipe.get("category"), set())
    return 1.0 if any(n not in seen for n in flavour_names(recipe)) else 0.0


def evaluate_population(recipes):
    """fitness = MAX_FITNESS * quality * (BASE + NOVELTY_WEIGHT*novelty + SURPRISE_WEIGHT*surprise).
    Multiplying (instead of adding) means weak baking can never be rescued
    by novelty, and there is no threshold cliff."""
    for r in recipes:
        q = evaluate_recipe(r)
        nov = novelty_score(r)
        r["quality_score"] = q * MAX_FITNESS
        r["novelty_score"] = nov
        sur = surprise_score(r)
        r["surprise_score"] = sur
        r["fitness"] = MAX_FITNESS * q * (BASE_WEIGHT + NOVELTY_WEIGHT * nov + SURPRISE_WEIGHT * sur)


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

    # the child takes category / meal type from the parent that contributed more ingredients
    major = r1 if len(r1a) >= len(r2b) else r2

    _recipe_counter[0] += 1
    new_recipe = {
        "name": f"recipe {_recipe_counter[0]}",
        "category": major.get("category", "cookie"),
        "meal_type": major.get("meal_type", "snack"),
        "ingredients": copy.deepcopy(r1a) + copy.deepcopy(r2b),
    }
    return new_recipe


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def _scaled_amount(value, unit):
    """Round an amount without truncating small values to 0 or 1.
    Whole items are rounded to an integer between 1 and MAX_UNIT_COUNT."""
    if unit == "unit":
        return max(1, min(MAX_UNIT_COUNT, round(value)))
    return max(0.5, round(value, 1)) if value < 10 else int(value)


def _present_amount(value, unit):
    """Round to a value a person would actually write in a recipe."""
    if unit == "unit":
        return max(1, min(MAX_UNIT_COUNT, round(value)))
    if value < 10:
        return max(0.5, round(value * 2) / 2)  # nearest 0.5
    if value < 20:
        return int(round(value))  # nearest 1
    return int(5 * round(value / 5))  # nearest 5


def round_for_presentation(recipe):
    """Applied once, to the final population only."""
    for i in recipe["ingredients"]:
        i["amount"] = _present_amount(i["amount"], i["unit"])


def mutate_recipe(recipe, all_ingredients, by_role, same_role_prob=0.7):
    op = random.randint(0, 4)
    ingredients = recipe["ingredients"]

    if op == 0 and ingredients:
        # change the amount of a random ingredient by +/- 10-30%
        i = random.randrange(len(ingredients))
        ingredients[i] = dict(ingredients[i])
        delta = random.uniform(-0.3, 0.3)
        ingredients[i]["amount"] = _scaled_amount(ingredients[i]["amount"] * (1 + delta), ingredients[i]["unit"])

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
        ingredients[i]["amount"] = choice["amount"]  # bug fix: amount used to be kept from the old ingredient

    elif op == 2:
        # add a new ingredient drawn from the inspiring set
        new_ing = dict(random.choice(all_ingredients))
        ingredients.append(new_ing)

    elif op == 3 and len(ingredients) > 1:
        # delete a random ingredient
        i = random.randrange(len(ingredients))
        ingredients.pop(i)

    elif op == 4:
        # transplant: bring in a mix-in that belongs to a *different* category
        # (e.g. bacon into a cookie), replacing an existing mix-in if there is one
        cat = recipe.get("category")
        foreign = [i for c, lst in MIXINS_BY_CATEGORY.items() if c != cat for i in lst]
        if foreign:
            new_ing = dict(random.choice(foreign))
            slots = [k for k, i in enumerate(ingredients) if i["role"] == "mix-in"]
            if slots:
                ingredients[random.choice(slots)] = new_ing
            else:
                ingredients.append(new_ing)


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

    # drop crumb-sized leftovers (e.g. 1g of cocoa in a 250g flour base); small-by-nature roles are kept
    total_structure = sum(i["amount"] for i in ingredients if i["role"] == "structure" and i["unit"] in ("g", "ml"))
    if total_structure > 0:
        ingredients = [i for i in ingredients
                       if i["unit"] not in ("g", "ml") or i["role"] in TRACE_ROLES
                       or i["amount"] >= MIN_SHARE * total_structure]

    structure_weight = sum(i["amount"] for i in ingredients if i["role"] == "structure" and i["unit"] in ("g", "ml"))
    if structure_weight > 0:
        scale = target_structure_weight / structure_weight
        for i in ingredients:
            if i["unit"] in ("g", "ml"):
                i["amount"] = _scaled_amount(i["amount"] * scale, i["unit"])
            elif i["unit"] == "unit":
                # round whole-item ingredients (eggs, bananas...) to the
                # nearest whole number, scaled gently, minimum of 1
                i["amount"] = _scaled_amount(i["amount"] * scale, i["unit"])

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
    """Identity used to detect seed copies: category + ingredient names
    (amounts and decorations ignored)."""
    return (recipe.get("category"), tuple(sorted(recipe_ingredient_set(recipe))))


def select_population(population, new_recipes, size):
    """Greedy diverse survivor selection, best fitness first:
      - inspiring-set recipes are never eligible,
      - a recipe is skipped if it is too close to one already kept,
      - a category cannot exceed MAX_CATEGORY_SHARE of the population,
      - if that leaves fewer than `size` survivors, skipped recipes top it up.
    """
    cap = max(2, int(MAX_CATEGORY_SHARE * size))
    pool = sorted(population + new_recipes, key=lambda r: r["fitness"], reverse=True)
    survivors, skipped, counts = [], [], defaultdict(int)
    for r in pool:
        if recipe_signature(r) in INSPIRING_SIGNATURES:
            continue
        cat = r.get("category")
        too_close = any(recipe_distance(r, s) < MIN_DISTANCE for s in survivors)
        if counts[cat] >= cap or too_close:
            skipped.append(r)
            continue
        survivors.append(r)
        counts[cat] += 1
        if len(survivors) == size:
            break
    if len(survivors) < size:
        survivors += skipped[: size - len(survivors)]
    survivors.sort(key=lambda r: r["fitness"], reverse=True)
    return survivors


@dataclass
class RunResult:
    best_fitness_history: list = field(default_factory=list)
    mean_fitness_history: list = field(default_factory=list)
    mean_quality_history: list = field(default_factory=list)
    mean_novelty_history: list = field(default_factory=list)
    final_population: list = field(default_factory=list)


def run_evolution(recipes, population_size=30, generations=100, seed=None):
    if seed is not None:
        random.seed(seed)

    learn_targets(recipes)  # must happen before any fitness evaluation

    global INSPIRING_RECIPES, INSPIRING_SIGNATURES
    INSPIRING_RECIPES = recipes
    INSPIRING_SIGNATURES = {recipe_signature(r) for r in recipes}

    all_ingredients = collect_all_ingredients(recipes)
    by_role = ingredients_by_role(all_ingredients)

    population = init_population(recipes, population_size)

    result = RunResult()
    for gen in range(generations):
        new_recipes = generate_recipes(population_size * 2, population, all_ingredients, by_role)
        population = select_population(population, new_recipes, population_size)
        result.best_fitness_history.append(population[0]["fitness"])
        result.mean_fitness_history.append(sum(r["fitness"] for r in population) / len(population))
        result.mean_quality_history.append(sum(r["quality_score"] for r in population) / len(population))
        result.mean_novelty_history.append(sum(r["novelty_score"] for r in population) / len(population))

    for r in population:
        round_for_presentation(r)
        if r["name"].startswith("recipe "):
            name_recipe(r)

    result.final_population = population
    return result
