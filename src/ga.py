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


# Settings
# --- What a valid recipe looks like
ESSENTIAL_ROLES = ["structure", "fat", "sweetener"]
MIN_INGREDIENTS = 4
MAX_INGREDIENTS = 14
TARGET_STRUCTURE_WEIGHT = 250  # grams of "structure" ingredients after normalisation
MAX_UNIT_COUNT = 6             # cap for whole-item ingredients (no recipe gets 10 eggs)
MIN_SHARE = 0.02               # g/ml ingredients below this share of the structure weight are dropped as crumbs...
TRACE_ROLES = ("seasoning", "leavening", "flavour")  # ...except roles that are small by nature
EGG_G = 50                     # assumed weight of an egg, so eggs count towards the binding ratio

# --- Quality score (0..1)
# How much each ratio (relative to flour weight) matters
FEATURE_WEIGHT = {
    "fat": 3, "sweetener": 3, "binding": 2, "liquid": 2,
    "leavening": 2, "salt": 1, "mix-in": 1.5, "flavour": 1,
}
FEATURES = list(FEATURE_WEIGHT)
RATIO_WEIGHT, PAIR_WEIGHT, MEAL_WEIGHT = 0.75, 0.15, 0.10  # how the three quality parts are blended
MISSING_ROLE_FACTOR = 0.3      # quality is multiplied by this for each missing essential role
SIZE_FACTOR = 0.7              # quality multiplier for too few / too many ingredients
MIN_CATEGORY_EXAMPLES = 5      # categories with fewer seed recipes use the global targets
SPREAD_FLOOR = (0.25, 0.01)    # (relative, absolute) minimum spread, so no ratio is impossibly strict

# --- Fitness = MAX_FITNESS x quality x (BASE + NOVELTY x novelty + SURPRISE x surprise)
MAX_FITNESS = 33
NOVELTY_WEIGHT = 0.2
SURPRISE_WEIGHT = 0.2
BASE_WEIGHT = 1 - NOVELTY_WEIGHT - SURPRISE_WEIGHT

# --- Survivor selection
MIN_DISTANCE = 0.15            # survivors must differ from each other by at least this Jaccard distance
MAX_CATEGORY_SHARE = 0.4       # no category may take more than this share of the population


# ---------------------------------------------------------------------------
# Knowledge learned from the inspiring set (filled by learn_from_inspiring_set)
# ---------------------------------------------------------------------------

TARGETS = {}                            # category -> feature -> (median, spread)
PAIRS = set()                           # (mix-in/flavour) pairs seen together in real recipes
MEAL_FLAVOURS = defaultdict(set)        # meal type -> mix-ins/flavours seen with it
CATEGORY_FLAVOURS = defaultdict(set)    # category -> mix-ins/flavours seen in it
MIXINS_BY_CATEGORY = defaultdict(list)  # category -> mix-in ingredient dicts (for transplants)
INSPIRING_RECIPES = []                  # novelty reference
INSPIRING_SIGNATURES = set()            # seed signatures that may not survive


def load_inspiring_set(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["recipes"], data.get("roles", {})


def collect_all_ingredients(recipes):
    """Every ingredient dict from every recipe in the inspiring set."""
    return [i for r in recipes for i in r["ingredients"]]


def ingredients_by_role(all_ingredients):
    """Group ingredient dicts by role, for role-aware mutation."""
    by_role = {}
    for i in all_ingredients:
        by_role.setdefault(i["role"], []).append(i)
    return by_role


def grams(i):
    """Approximate weight of an ingredient line (eggs count as EGG_G each)."""
    if i["unit"] in ("g", "ml"):
        return i["amount"]
    if i["unit"] == "unit" and "egg" in i["ingredient"].lower():
        return i["amount"] * EGG_G
    return 0


def flavour_names(recipe):
    """Sorted mix-in/flavour ingredient names (salt excluded)."""
    return sorted({i["ingredient"] for i in recipe["ingredients"]
                   if i["role"] in ("mix-in", "flavour") and i["ingredient"].lower() != "salt"})


def ratios(recipe):
    """Every feature as a ratio to the structure (flour) weight, or None if
    the recipe has no structural ingredient."""
    structure = sum(grams(i) for i in recipe["ingredients"] if i["role"] == "structure")
    if structure <= 0:
        return None
    out = {f: 0.0 for f in FEATURES}
    for i in recipe["ingredients"]:
        feature = "salt" if i["ingredient"].lower() == "salt" else i["role"]
        if feature in out:
            out[feature] += grams(i) / structure
    return out


def recipe_ingredient_set(recipe):
    """Ingredient names, ignoring amounts and decorations (decorations have
    near-unique names, so counting them would reward padding)."""
    return {i["ingredient"] for i in recipe["ingredients"] if i["role"] != "decoration"}


def recipe_signature(recipe):
    """Identity used to detect seed copies: category + ingredient names."""
    return (recipe.get("category"), tuple(sorted(recipe_ingredient_set(recipe))))


def _ratio_stats(rows):
    """Per-feature (median, spread) over a list of ratio dicts."""
    rel_floor, abs_floor = SPREAD_FLOOR
    stats = {}
    for f in FEATURES:
        values = [row[f] for row in rows]
        median = statistics.median(values)
        mad = statistics.median(abs(v - median) for v in values) * 1.4826
        stats[f] = (median, max(mad, rel_floor * median, abs_floor))
    return stats


def learn_from_inspiring_set(recipes):
    """Fill all the knowledge tables above from the inspiring set."""
    for table in (PAIRS, MEAL_FLAVOURS, CATEGORY_FLAVOURS, MIXINS_BY_CATEGORY, TARGETS):
        table.clear()
    INSPIRING_RECIPES[:] = recipes
    INSPIRING_SIGNATURES.clear()
    INSPIRING_SIGNATURES.update(recipe_signature(r) for r in recipes)

    ratio_rows = defaultdict(list)
    for r in recipes:
        category = r.get("category")
        rt = ratios(r)
        if rt:
            ratio_rows[category].append(rt)
        names = flavour_names(r)
        PAIRS.update(combinations(names, 2))
        MEAL_FLAVOURS[r.get("meal_type")].update(names)
        CATEGORY_FLAVOURS[category].update(names)
        MIXINS_BY_CATEGORY[category].extend(dict(i) for i in r["ingredients"] if i["role"] == "mix-in")

    TARGETS["_all"] = _ratio_stats([row for rows in ratio_rows.values() for row in rows])
    for category, rows in ratio_rows.items():
        TARGETS[category] = _ratio_stats(rows) if len(rows) >= MIN_CATEGORY_EXAMPLES else TARGETS["_all"]


# ---------------------------------------------------------------------------
# Fitness
# ---------------------------------------------------------------------------

def ratio_score(recipe):
    """0..1: how close every ratio is to the typical value for the recipe's
    category (smooth Gaussian, so there is always a gradient to climb)."""
    rt = ratios(recipe)
    if rt is None:
        return 0.0
    target = TARGETS.get(recipe.get("category"), TARGETS["_all"])
    total = 0.0
    for f, weight in FEATURE_WEIGHT.items():
        median, spread = target[f]
        z = (rt[f] - median) / spread
        total += weight * math.exp(-0.5 * z * z)
    return total / sum(FEATURE_WEIGHT.values())


def pair_score(recipe):
    """Fraction of mix-in/flavour pairs that appear together in real recipes."""
    pairs = list(combinations(flavour_names(recipe), 2))
    return sum(p in PAIRS for p in pairs) / len(pairs) if pairs else 1.0


def meal_score(recipe):
    """Fraction of mix-ins/flavours seen in real recipes of the same meal type."""
    names = flavour_names(recipe)
    seen = MEAL_FLAVOURS.get(recipe.get("meal_type"), set())
    return sum(n in seen for n in names) / len(names) if names else 1.0


def evaluate_recipe(recipe):
    """Quality in 0..1. Role coverage and recipe size are multiplicative
    gates, so there are no cliffs."""
    if not TARGETS:
        raise RuntimeError("Call learn_from_inspiring_set(recipes) before evaluating recipes.")
    quality = (RATIO_WEIGHT * ratio_score(recipe)
               + PAIR_WEIGHT * pair_score(recipe)
               + MEAL_WEIGHT * meal_score(recipe))

    roles_present = {i["role"] for i in recipe["ingredients"]}
    quality *= MISSING_ROLE_FACTOR ** sum(r not in roles_present for r in ESSENTIAL_ROLES)

    if not MIN_INGREDIENTS <= len(recipe["ingredients"]) <= MAX_INGREDIENTS:
        quality *= SIZE_FACTOR
    return quality


def recipe_distance(r1, r2):
    """Jaccard distance between the ingredient-name sets of two recipes."""
    s1, s2 = recipe_ingredient_set(r1), recipe_ingredient_set(r2)
    if not s1 and not s2:
        return 0.0
    return 1.0 - len(s1 & s2) / len(s1 | s2)


def novelty_score(recipe):
    """Distance to the most similar recipe in the inspiring set."""
    if not INSPIRING_RECIPES:
        return 0.0
    return min(recipe_distance(recipe, r) for r in INSPIRING_RECIPES)


def surprise_score(recipe):
    """1 if the recipe has a mix-in/flavour never seen in its own category,
    else 0. Capped: one deliberate surprise counts, stacking several doesn't."""
    seen = CATEGORY_FLAVOURS.get(recipe.get("category"), set())
    return 1.0 if any(n not in seen for n in flavour_names(recipe)) else 0.0


def evaluate_population(recipes):
    """Store quality, novelty, surprise and the combined fitness on each recipe."""
    for r in recipes:
        quality = evaluate_recipe(r)
        r["quality_score"] = quality * MAX_FITNESS
        r["novelty_score"] = novelty_score(r)
        r["surprise_score"] = surprise_score(r)
        r["fitness"] = MAX_FITNESS * quality * (
            BASE_WEIGHT + NOVELTY_WEIGHT * r["novelty_score"] + SURPRISE_WEIGHT * r["surprise_score"])


# ---------------------------------------------------------------------------
# Selection and crossover
# ---------------------------------------------------------------------------

def select_recipe(population):
    """Roulette-wheel selection (fitness is never negative)."""
    return random.choices(population, weights=[r["fitness"] + 1e-9 for r in population])[0]


_recipe_counter = [0]


def crossover_recipes(r1, r2):
    """One-point crossover. The child takes its category and meal type from
    the parent that contributed more ingredients."""
    head = r1["ingredients"][:random.randint(0, len(r1["ingredients"]))]
    tail = r2["ingredients"][random.randint(0, len(r2["ingredients"])):]
    major = r1 if len(head) >= len(tail) else r2

    _recipe_counter[0] += 1
    return {
        "name": f"recipe {_recipe_counter[0]}",
        "category": major.get("category", "cookie"),
        "meal_type": major.get("meal_type", "snack"),
        "ingredients": copy.deepcopy(head) + copy.deepcopy(tail),
    }


# ---------------------------------------------------------------------------
# Amounts
# ---------------------------------------------------------------------------

def _scaled_amount(value, unit):
    """Round an amount without truncating small values to 0 or 1.
    Whole items become an integer between 1 and MAX_UNIT_COUNT."""
    if unit == "unit":
        return max(1, min(MAX_UNIT_COUNT, round(value)))
    return max(0.5, round(value, 1)) if value < 10 else int(value)


def _present_amount(value, unit):
    """Round to a value a person would actually write in a recipe."""
    if unit == "unit":
        return _scaled_amount(value, unit)
    if value < 10:
        return max(0.5, round(value * 2) / 2)  # nearest 0.5
    if value < 20:
        return int(round(value))               # nearest 1
    return int(5 * round(value / 5))           # nearest 5


def round_for_presentation(recipe):
    """Applied once, to the final population only."""
    for i in recipe["ingredients"]:
        i["amount"] = _present_amount(i["amount"], i["unit"])


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def mutate_recipe(recipe, all_ingredients, by_role, same_role_prob=0.7):
    """Apply one random mutation to a (freshly crossed-over) recipe."""
    ingredients = recipe["ingredients"]
    op = random.choice(("amount", "swap", "add", "delete", "transplant"))

    if op == "add":
        # add a new ingredient drawn from the inspiring set
        ingredients.append(dict(random.choice(all_ingredients)))

    elif op == "transplant":
        # bring in a mix-in from a *different* category (e.g. bacon into a
        # cookie), replacing an existing mix-in if there is one
        foreign = [i for cat, mixins in MIXINS_BY_CATEGORY.items()
                   if cat != recipe.get("category") for i in mixins]
        if foreign:
            new_ing = dict(random.choice(foreign))
            slots = [k for k, i in enumerate(ingredients) if i["role"] == "mix-in"]
            if slots:
                ingredients[random.choice(slots)] = new_ing
            else:
                ingredients.append(new_ing)

    elif ingredients:  # the remaining operations need an existing ingredient
        k = random.randrange(len(ingredients))
        ing = ingredients[k]

        if op == "amount":
            # change the amount by up to +/- 30%
            ing["amount"] = _scaled_amount(ing["amount"] * (1 + random.uniform(-0.3, 0.3)), ing["unit"])

        elif op == "swap":
            # swap for another ingredient; usually one with the same role to
            # keep the recipe coherent, sometimes any ingredient for novelty
            pool = by_role.get(ing["role"]) if random.random() < same_role_prob else None
            choice = random.choice(pool or all_ingredients)
            ing.update(ingredient=choice["ingredient"], role=choice["role"],
                       unit=choice["unit"], amount=choice["amount"])

        elif op == "delete" and len(ingredients) > 1:
            ingredients.pop(k)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _structure_weight(ingredients):
    return sum(i["amount"] for i in ingredients if i["role"] == "structure" and i["unit"] in ("g", "ml"))


def normalise_recipe(recipe, target_structure_weight=TARGET_STRUCTURE_WEIGHT):
    """Baking-appropriate normalisation. Ratios relative to the flour (the
    "structure" ingredients) are what matter in baking, so instead of scaling
    to a fixed total volume (as Simple PIERRE did for soup) this:
      1. merges duplicate ingredient entries,
      2. drops crumb-sized leftovers (e.g. 1g of cocoa next to 250g of flour),
      3. scales everything so the structure weight equals the target;
         whole items (eggs, bananas...) are rounded to whole numbers.
    """
    merged = {}
    for i in recipe["ingredients"]:
        key = (i["ingredient"], i["role"])
        if key in merged:
            merged[key]["amount"] += i["amount"]
        else:
            merged[key] = dict(i)
    ingredients = list(merged.values())

    structure = _structure_weight(ingredients)
    if structure > 0:
        ingredients = [i for i in ingredients
                       if i["unit"] == "unit" or i["role"] in TRACE_ROLES
                       or i["amount"] >= MIN_SHARE * structure]

    structure = _structure_weight(ingredients)
    if structure > 0:
        scale = target_structure_weight / structure
        for i in ingredients:
            i["amount"] = _scaled_amount(i["amount"] * scale, i["unit"])

    recipe["ingredients"] = ingredients


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

ADJECTIVES = ["Golden", "Chewy", "Crumbly", "Spiced", "Rustic", "Decadent",
              "Hearty", "Fluffy", "Buttery", "Caramelised"]


def name_recipe(recipe):
    """Descriptive name from the dominant mix-in/flavour ingredient."""
    adjective = random.choice(ADJECTIVES)
    category = recipe.get("category", "Bake").replace("_", " ").title()
    candidates = [i for i in recipe["ingredients"] if i["role"] in ("mix-in", "flavour")]
    if candidates:
        top = max(candidates, key=lambda i: i["amount"])
        recipe["name"] = f"{adjective} {top['ingredient'].title()} {category}"
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
    """Create `size` evaluated children by selection, crossover, mutation and normalisation."""
    children = []
    while len(children) < size:
        child = crossover_recipes(select_recipe(population), select_recipe(population))
        mutate_recipe(child, all_ingredients, by_role)
        normalise_recipe(child)
        children.append(child)
    evaluate_population(children)
    return children


def select_population(population, children, size):
    """Greedy diverse survivor selection, best fitness first:
      - inspiring-set recipes are never eligible,
      - a recipe is skipped if it is too close to one already kept,
      - a category cannot exceed MAX_CATEGORY_SHARE of the population,
      - if that leaves fewer than `size` survivors, skipped recipes top it up.
    """
    cap = max(2, int(MAX_CATEGORY_SHARE * size))
    pool = sorted(population + children, key=lambda r: r["fitness"], reverse=True)
    survivors, skipped, counts = [], [], defaultdict(int)
    for r in pool:
        if recipe_signature(r) in INSPIRING_SIGNATURES:
            continue
        category = r.get("category")
        too_close = any(recipe_distance(r, s) < MIN_DISTANCE for s in survivors)
        if counts[category] >= cap or too_close:
            skipped.append(r)
            continue
        survivors.append(r)
        counts[category] += 1
        if len(survivors) == size:
            break
    survivors += skipped[: size - len(survivors)]
    survivors.sort(key=lambda r: r["fitness"], reverse=True)
    return survivors


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _mean(population, key):
    return sum(r[key] for r in population) / len(population)


@dataclass
class RunResult:
    best_fitness_history: list = field(default_factory=list)
    mean_fitness_history: list = field(default_factory=list)
    mean_quality_history: list = field(default_factory=list)
    mean_novelty_history: list = field(default_factory=list)
    final_population: list = field(default_factory=list)

    def record(self, population):
        self.best_fitness_history.append(population[0]["fitness"])
        self.mean_fitness_history.append(_mean(population, "fitness"))
        self.mean_quality_history.append(_mean(population, "quality_score"))
        self.mean_novelty_history.append(_mean(population, "novelty_score"))


def run_evolution(recipes, population_size=30, generations=100, seed=None):
    if seed is not None:
        random.seed(seed)

    learn_from_inspiring_set(recipes)  # must happen before any fitness evaluation
    all_ingredients = collect_all_ingredients(recipes)
    by_role = ingredients_by_role(all_ingredients)

    population = init_population(recipes, population_size)
    result = RunResult()
    for _ in range(generations):
        children = generate_recipes(population_size * 2, population, all_ingredients, by_role)
        population = select_population(population, children, population_size)
        result.record(population)

    for r in population:
        round_for_presentation(r)
        if r["name"].startswith("recipe "):
            name_recipe(r)

    result.final_population = population
    return result
