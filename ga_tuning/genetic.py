"""精英保留、排名轮盘赌、两点交叉、逐位变异。GA 只通过 evaluate 获取成绩。"""
import random
from statistics import mean

from .encoding import BITS, GRID, decode, encode, random_chromosome, repair


def rates(stale):
    # 每连续两代没有刷新历史冠军，增大三个参数；改善后由调用方清零。
    level = min(stale // 2, 4)
    return 1.0 + 0.25 * level, 0.70 + 0.06 * level, (1 + level) / BITS


def child(ranked, rng, power, p_cross, p_mutate):
    # 排名越靠前，(种群大小 - 名次 + 1)^power 越大，适用于最小化得分。
    weights = [r ** power for r in range(len(ranked), 0, -1)]
    a, b = rng.choices(ranked, weights=weights, k=2)
    bits = a
    if rng.random() < p_cross:
        left, right = sorted(rng.sample(range(BITS + 1), 2))
        bits = a[:left] + b[left:right] + a[right:]
    mutated = []
    for bit in bits:
        if rng.random() < p_mutate:
            bit = "1" if bit == "0" else "0"
        mutated.append(bit)
    return repair("".join(mutated), rng)


def initial_population(default, size, rng):
    # 真实默认值可以在搜索范围外；只有参与遗传操作的个体才需要可编码。
    anchor = {name: min(levels, key=lambda v: abs(v - default[name]))
              for name, levels in GRID.items()}
    counts = GRID["init_points"]
    population = [encode(anchor), encode(dict(anchor, init_points=counts[0]))]
    population = list(dict.fromkeys(population))
    slots = size - len(population)
    for i in range(slots):
        initial = counts[round((len(counts) - 1) * i / max(1, slots - 1))]
        while True:
            recipe = decode(random_chromosome(rng))
            recipe["init_points"] = initial
            bits = encode(recipe)
            if bits not in population:
                population.append(bits)
                break
    return population


def evolve(settings, seed, default, evaluate):
    rng = random.Random(seed)
    # 第 0 代只测试实际 BO 默认参数，不交叉、不变异，也不把超范围值截断。
    baseline = evaluate("default", default, 0)
    history = [dict(generation=0, champion="default", mean_loss=baseline["loss"],
                    population_mean_loss=baseline["loss"], improvement=0.0,
                    mean_base_loss=baseline["base_loss"], failure_rate=baseline["patience_failed"],
                    stale=0, selection_power=None, crossover=None, bit_mutation=None, new_recipes=1)]
    population = initial_population(default, settings.population, rng)
    scores, members = {}, []
    best_loss, stale = float("inf"), 0
    for generation in range(1, settings.generations + 1):
        fresh = [bits for bits in population if bits not in scores]
        for bits in fresh:
            scores[bits] = evaluate(bits, decode(bits), generation)
        ranked = sorted(population, key=lambda bits: scores[bits]["loss"])
        champion = ranked[0]
        score = scores[champion]
        loss = score["loss"]
        stale = 0 if loss < best_loss else stale + 1
        best_loss = loss  # 精英保留，上一代冠军参加本代排名。
        power, crossover, mutation = rates(stale)
        history.append(dict(generation=generation, champion=champion, mean_loss=loss,
                            population_mean_loss=mean(scores[b]["loss"] for b in ranked),
                            improvement=baseline["loss"] - loss, mean_base_loss=score["base_loss"],
                            failure_rate=score["patience_failed"], stale=stale,
                            selection_power=power, crossover=crossover, bit_mutation=mutation,
                            new_recipes=len(fresh)))
        members.extend(dict(generation=generation, rank=i, chromosome=bits, loss=scores[bits]["loss"])
                       for i, bits in enumerate(ranked, 1))
        print(f"第 {generation} 代完成，最佳 loss={loss:.6f}，较默认降低 {baseline['loss'] - loss:+.6f}",
              flush=True)
        if generation < settings.generations:
            population = [champion]
            while len(population) < settings.population:
                bits = child(ranked, rng, power, crossover, mutation)
                if bits not in population:
                    population.append(bits)
    return champion, history, members
