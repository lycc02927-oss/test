"""精英保留、排名轮盘赌、两点交叉、逐位变异。GA 只通过 evaluate 获取成绩。"""
import random

from .encoding import BASELINE, BITS, GRID, decode, encode, random_chromosome, repair


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


def evolve(plan, seed, evaluate):
    rng = random.Random(seed)
    population = [encode(BASELINE)]
    if plan.spread_initial:
        # 先以相同其余参数测试最少初始点，再让随机配方覆盖初始点的整个范围。
        # balanced 首代的六个点数为 30、10、10、23、37、50。
        values = GRID["init_points"]
        population.append(encode(dict(BASELINE, init_points=values[0])))
        slots = plan.population - len(population)
        for i in range(slots):
            index = round(i * (len(values) - 1) / max(1, slots - 1))
            recipe = decode(random_chromosome(rng))
            recipe["init_points"] = values[index]
            bits = encode(recipe)
            if bits not in population:
                population.append(bits)
    while len(population) < plan.population:
        bits = random_chromosome(rng)
        if bits not in population:
            population.append(bits)

    history, members, scores = [], [], {}
    champion = None
    stale = 0
    for generation in range(1, plan.generations + 1):
        fresh = [bits for bits in population if bits not in scores]
        scores.update(evaluate(fresh))
        ranked = sorted(population, key=scores.get)
        if champion is None or scores[ranked[0]] < scores[champion]:
            champion, stale = ranked[0], 0
        else:
            stale += 1
        power, crossover, mutation = rates(stale)
        score = scores[champion]
        history.append(dict(generation=generation, champion=champion, stale=stale,
                            selection_power=power, crossover=crossover, bit_mutation=mutation,
                            new_recipes=len(fresh), worst_failure_rate=score.worst_failure_rate,
                            mean_failure_rate=score.mean_failure_rate, mean_loss=score.mean_loss))
        for rank, bits in enumerate(ranked, 1):
            members.append(dict(generation=generation, rank=rank, chromosome=bits))
        print(f"第 {generation} 代完成，最弱工况失败率 {score.worst_failure_rate:.1%}，平均分 {score.mean_loss:.4f}", flush=True)
        if generation == plan.generations:
            break
        # 精英原样保留。其余名额从父代产生，同代去重避免重复测量。
        population = [champion]
        while len(population) < plan.population:
            bits = child(ranked, rng, power, crossover, mutation)
            if bits not in population:
                population.append(bits)
    return champion, history, members
