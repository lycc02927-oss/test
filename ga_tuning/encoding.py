"""每段二进制存一个参数的档位序号；顺序固定，方便从 Excel 还原。"""
from itertools import accumulate

RESTARTS = (1, *range(10, 101, 10))
GRID = {
    "sigma3_rel": tuple(i / 100 for i in range(101)),
    "n_restarts": RESTARTS,
    "n_restarts_ei": RESTARTS,
    "xi_init": tuple(i / 1000 for i in range(5, 101)),
    "xi_final": tuple(i / 1000 for i in range(11)),
    "global_prob": tuple(i / 10 for i in range(11)),
    "ls_lo": tuple(range(1, 101)),
    "ls_hi": tuple(range(100, 301)),
    "init_points": tuple(range(10, 51)),
}
BASELINE = dict(zip(GRID, (0.01, 10, 10, 0.01, 0, 0, 40, 300, 30)))
WIDTH = {key: (len(values) - 1).bit_length() for key, values in GRID.items()}
ENDS = tuple(accumulate(WIDTH.values()))
BITS = ENDS[-1]


def encode(recipe):
    pieces = []
    for key, values in GRID.items():
        index = values.index(recipe[key])
        pieces.append(format(index, f"0{WIDTH[key]}b"))
    return "".join(pieces)


def decode(bits):
    recipe = {}
    start = 0
    for (key, values), end in zip(GRID.items(), ENDS):
        recipe[key] = values[int(bits[start:end], 2)]
        start = end
    return recipe


def repair(bits, rng):
    """交叉后可能出现不存在的档位，仅重抽该基因，避免大量编码挤到上界。"""
    recipe = {}
    start = 0
    for (key, values), end in zip(GRID.items(), ENDS):
        index = int(bits[start:end], 2)
        recipe[key] = values[index] if index < len(values) else rng.choice(values)
        start = end
    # 保持 EI 衰减、长度尺度有正的优化区间。仅修复违反关系的参数。
    if recipe["xi_final"] > recipe["xi_init"]:
        legal = [v for v in GRID["xi_final"] if v <= recipe["xi_init"]]
        recipe["xi_final"] = rng.choice(legal)
    if recipe["ls_hi"] == recipe["ls_lo"]:
        recipe["ls_hi"] = rng.choice(GRID["ls_hi"][1:])
    return encode(recipe)


def random_chromosome(rng):
    recipe = {key: rng.choice(values) for key, values in GRID.items()}
    return repair(encode(recipe), rng)
