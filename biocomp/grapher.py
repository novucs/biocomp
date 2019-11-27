import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


class Log:
    def __init__(self, settings, entries, name):
        self.settings = settings
        self.entries = entries
        self.name = name


def load_logfiles(names):
    logfiles = []

    for name in names:
        with open(name, "r") as f:
            lines = f.readlines()

        settings = dict(l.strip().split(':', maxsplit=1) for l in lines if l.startswith('\t'))
        entries = [{
            v.split(':')[0]: v.split(":", maxsplit=1)[1]
            for v in l.split(" ")
            if ':' in v
        } for l in lines if "generation" in l]

        if 'rule_count' not in settings:
            settings['rule_count'] = 1

        for entry in entries:
            if 'rule_count' not in entry:
                entry['rule_count'] = 0

        context = next((
            l.replace('CreationContext(', '').replace(')', '')
            for l in lines if 'Context' in l
        ), None)
        if context:
            settings.update({
                v.split('=')[0]: v.split('=', maxsplit=1)[1]
                for v in context.split(', ')
            })

        if len(entries) == 0:
            print('Empty log:', name)
            continue

        logfiles.append(Log(settings, entries, name))

    return logfiles


def load_tree_logfiles(names):
    logfiles = []

    for name in names:
        with open(name, "r") as f:
            lines = f.readlines()

        settings = {'rule_count': 1}
        entries = [{v.split(':')[0]: v.split(':')[1].strip() for v in l.split('\t')} for l in lines]

        for entry in entries:
            entry['rule_count'] = 0

        if len(entries) == 0:
            print('Empty log:', name)
            continue

        logfiles.append(Log(settings, entries, name))

    return logfiles


class AreaPlot:
    def __init__(self, run_count, generation_count):
        self.run_count = run_count
        self.total_generations = generation_count
        self.mean = [0] * generation_count
        self.error_max = [-float('inf')] * generation_count
        self.error_min = [float('inf')] * generation_count
        self.generation = 0

    def reset(self):
        self.generation = 0

    def update(self, fitness):
        if self.generation >= self.total_generations:
            raise ValueError(f'Unable to add new generation fitness when cap is '
                             f'already reached: {self.generation} / {self.total_generations}')

        self.mean[self.generation] += fitness / self.run_count
        self.error_max[self.generation] = max(fitness, self.error_max[self.generation])
        self.error_min[self.generation] = min(fitness, self.error_min[self.generation])
        self.generation += 1


def plot_fitness_area(generation_count, logs, key, line_args=None, fill_args=None):
    line_args = line_args or {}
    fill_args = fill_args or {}

    generations = list(range(generation_count))
    plotter = AreaPlot(len(logs), generation_count)

    for log in logs:
        plotter.reset()

        if len(log.entries) == 0:
            raise ValueError("Log must not be empty:", log.name)

        for generation, entry in enumerate(log.entries):
            if generation == generation_count:
                break

            rule_count = float(entry['rule_count'])
            base_fitness = float(log.settings['rule_count']) - rule_count
            fitness = (float(entry[key]) * base_fitness) / float(log.settings['rule_count'])
            plotter.update(fitness)

        reached = generation

        for generation in range(reached, generation_count - 1):
            plotter.update(fitness)

    plt.plot(
        generations, plotter.mean,
        label=key,
        **line_args,
    )
    plt.fill_between(
        generations, plotter.error_min, plotter.error_max,
        alpha=0.5,
        linewidth=0,
        **fill_args,
    )
    axes = plt.gca()
    axes.set_ylim([0, 1])
    axes.set_xlim([0, generation_count])


def plot_parameter_sweep(title, logs):
    crossovers = sorted({float(l.settings['crossover_rate']) for l in logs})
    mutations = sorted({float(l.settings['mutation_rate']) for l in logs})

    xs = crossovers
    ys = mutations
    zs1 = [[0. for _ in range(len(crossovers))] for _ in range(len(mutations))]
    zs2 = [[0. for _ in range(len(crossovers))] for _ in range(len(mutations))]

    power = 1
    values = [(
        crossovers.index(float(log.settings['crossover_rate'])),
        mutations.index(float(log.settings['mutation_rate'])),
        (sum(float(entry['train_fitness_mean']) for entry in log.entries) / len(log.entries)) ** power,
        (sum(float(entry['train_fitness_best']) for entry in log.entries) / len(log.entries)) ** power,
    ) for log in logs]

    for x, y, z1, z2 in values:
        sample_count = sum(1 for i, j, *_ in values if i == x and j == y)
        zs1[y][x] += z1 / sample_count
        zs2[y][x] += z2 / sample_count

    # # Reverse order of columns to get a different display angle
    # ys = list(reversed(ys))
    # zs1 = list(reversed(zs1))
    # zs2 = list(reversed(zs2))

    data1 = np.array(zs1)
    data2 = np.array(zs2)

    fig = plt.figure()
    ax: Axes3D = fig.gca(projection='3d')

    lx = len(data1[0])  # Work out matrix dimensions
    ly = len(data1[:, 0])
    xpos = np.arange(0, lx, 1)  # Set up a mesh of positions
    ypos = np.arange(0, ly, 1)
    xpos, ypos = np.meshgrid(xpos + 0.25, ypos + 0.25)

    xpos = xpos.flatten()  # Convert positions to 1D array
    ypos = ypos.flatten()
    zpos = np.zeros(lx * ly)

    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dz1 = data1.flatten()
    dz2 = data2.flatten()

    for k in range(len(xpos)):
        ax.bar3d(xpos[k], ypos[k], zpos[k], dx[k], dy[k], dz1[k], color='C1', alpha=1)
        ax.bar3d(xpos[k], ypos[k], zpos[k] + dz1[k], dx[k], dy[k], dz2[k] - dz1[k], color='C0', alpha=1)

    ax.set_zlim([0, 1])

    ax.set_xticks([i + .5 for i in range(len(xs))])
    ax.set_xticklabels(xs)

    ax.set_yticks([i + .5 for i in range(len(ys))])
    ax.set_yticklabels(ys)

    ax.set_xlabel('Crossover Rate')
    ax.set_ylabel('Mutation Rate')
    ax.set_zlabel('Fitness')

    best_proxy = plt.Rectangle((0, 0), 1, 1, fc="C0")
    mean_proxy = plt.Rectangle((0, 0), 1, 1, fc="C1")
    ax.legend([best_proxy, mean_proxy], ['best fitness', 'mean fitness'], loc='lower left')
    plt.title(title)

    plt.show()


def main():
    experiments = {
        # # rule_count:60
        # # population_size: 100
        # # crossover_chance: 0.85
        # # mutation_rate: 1
        # # selection_switch_threshold: 0.1
        # # covered_best_variations: 5
        # # tournament_size: 5
        # # distill_inheritance_chance: 0.33
        # # cover_chance: 0.1
        # # fitness_threshold: 1
        # 'dynamic_mutation': load_logfiles([
        #     "c/logs/data1/2019-11-22.05:30:55.log",
        #     "c/logs/data1/2019-11-22.05:31:31.log",
        #     "c/logs/data1/2019-11-22.05:32:06.log",
        #     "c/logs/data1/2019-11-22.05:32:30.log",
        #     "c/logs/data1/2019-11-22.05:35:27.log",
        #     "c/logs/data1/2019-11-22.05:36:01.log",
        #     "c/logs/data1/2019-11-22.05:36:49.log",
        #     "c/logs/data1/2019-11-22.05:37:32.log",
        #     "c/logs/data1/2019-11-22.05:38:35.log",
        #     "c/logs/data1/2019-11-22.05:39:12.log",
        #
        #     # "c/logs/data1/2019-11-22.06:49:03.log",
        #     # "c/logs/data1/2019-11-22.06:49:13.log",
        #     # "c/logs/data1/2019-11-22.06:49:25.log",
        #     # "c/logs/data1/2019-11-22.06:49:36.log",
        #     # "c/logs/data1/2019-11-22.06:49:46.log",
        #     # "c/logs/data1/2019-11-22.06:50:00.log",
        #     # "c/logs/data1/2019-11-22.06:50:11.log",
        #     # "c/logs/data1/2019-11-22.06:50:23.log",
        #     # "c/logs/data1/2019-11-22.06:50:35.log",
        #     # "c/logs/data1/2019-11-22.06:50:46.log",
        # ]),
        #
        # # rule_count:60
        # # population_size: 100
        # # crossover_chance: 0.85
        # # mutation_rate: 0.003
        # # selection_switch_threshold: 0.1
        # # covered_best_variations: 5
        # # tournament_size: 5
        # # distill_inheritance_chance: 0.33
        # # cover_chance: 0.1
        # # fitness_threshold: 1
        # 'static_mutation': load_logfiles([
        #     "c/logs/data1/2019-11-22.07:07:02.log",
        #     "c/logs/data1/2019-11-22.07:07:14.log",
        #     "c/logs/data1/2019-11-22.07:07:28.log",
        #     "c/logs/data1/2019-11-22.07:07:40.log",
        #     "c/logs/data1/2019-11-22.07:07:52.log",
        #     "c/logs/data1/2019-11-22.07:08:03.log",
        #     "c/logs/data1/2019-11-22.07:08:17.log",
        #     "c/logs/data1/2019-11-22.07:08:30.log",
        #     "c/logs/data1/2019-11-22.07:08:42.log",
        #     "c/logs/data1/2019-11-22.07:08:55.log",
        #
        #     # "c/logs/data1/2019-11-22.07:46:00.log",
        #     # "c/logs/data1/2019-11-22.07:46:11.log",
        #     # "c/logs/data1/2019-11-22.07:46:23.log",
        #     # "c/logs/data1/2019-11-22.07:46:36.log",
        #     # "c/logs/data1/2019-11-22.07:46:49.log",
        #     # "c/logs/data1/2019-11-22.07:47:01.log",
        #     # "c/logs/data1/2019-11-22.07:47:12.log",
        #     # "c/logs/data1/2019-11-22.07:47:25.log",
        #     # "c/logs/data1/2019-11-22.07:47:37.log",
        #     # "c/logs/data1/2019-11-22.07:47:48.log",
        # ]),
        #
        # # rule_count:5
        # # population_size: 100
        # # crossover_chance: 0.85
        # # mutation_rate: 0.003
        # # selection_switch_threshold: 0.1
        # # covered_best_variations: 5
        # # tournament_size: 5
        # # distill_inheritance_chance: 0.33
        # # cover_chance: 0.1
        # # fitness_threshold: 1.66667
        # 'static_rule_count': load_logfiles([
        #     "c/logs/data1/2019-11-22.07:17:15.log",
        #     "c/logs/data1/2019-11-22.07:17:22.log",
        #     "c/logs/data1/2019-11-22.07:17:29.log",
        #     "c/logs/data1/2019-11-22.07:17:36.log",
        #     "c/logs/data1/2019-11-22.07:17:43.log",
        #     "c/logs/data1/2019-11-22.07:17:50.log",
        #     "c/logs/data1/2019-11-22.07:17:57.log",
        #     "c/logs/data1/2019-11-22.07:18:04.log",
        #     "c/logs/data1/2019-11-22.07:18:11.log",
        #     "c/logs/data1/2019-11-22.07:18:18.log",
        # ]),
        #
        # # rule_count:60
        # # population_size: 100
        # # crossover_chance: 0.85
        # # mutation_rate: 0.003
        # # use_tournament_selection: false
        # # selection_switch_threshold: 0.1
        # # covered_best_variations: 5
        # # tournament_size: 5
        # # distill_inheritance_chance: 0.33
        # # cover_chance: 0.1
        # # fitness_threshold: 1
        # 'roulette_wheel_selection': load_logfiles([
        #     "c/logs/data1/2019-11-22.08:05:56.log",
        #     "c/logs/data1/2019-11-22.08:06:14.log",
        #     "c/logs/data1/2019-11-22.08:06:35.log",
        #     "c/logs/data1/2019-11-22.08:06:59.log",
        #     "c/logs/data1/2019-11-22.08:07:20.log",
        #     "c/logs/data1/2019-11-22.08:07:43.log",
        #     "c/logs/data1/2019-11-22.08:08:05.log",
        #     "c/logs/data1/2019-11-22.08:08:27.log",
        #     "c/logs/data1/2019-11-22.08:08:51.log",
        #     "c/logs/data1/2019-11-22.08:09:13.log",
        # ]),
        #
        # # rule_count: 60
        # # population_size: 100
        # # crossover_chance: 0.85
        # # mutation_rate: 0.01
        # # use_tournament_selection: 1
        # # selection_switch_threshold: 0.1
        # # covered_best_variations: 5
        # # tournament_size: 5
        # # distill_inheritance_chance: 0.33
        # # cover_chance: 0.1
        # # fitness_threshold: 1
        # 'boosted_mutation': load_logfiles([
        #     "c/logs/data1/2019-11-22.08:18:21.log",
        #     "c/logs/data1/2019-11-22.08:18:36.log",
        #     "c/logs/data1/2019-11-22.08:18:47.log",
        #     "c/logs/data1/2019-11-22.08:19:05.log",
        #     "c/logs/data1/2019-11-22.08:19:22.log",
        #     "c/logs/data1/2019-11-22.08:19:34.log",
        #     "c/logs/data1/2019-11-22.08:19:47.log",
        #     "c/logs/data1/2019-11-22.08:20:02.log",
        #     "c/logs/data1/2019-11-22.08:20:18.log",
        #     "c/logs/data1/2019-11-22.08:20:37.log",
        # ]),
        #
        # # rule_count: 60
        # # population_size: 100
        # # crossover_chance: 0.5
        # # mutation_rate: 0.003
        # # use_tournament_selection: 1
        # # selection_switch_threshold: 0.1
        # # covered_best_variations: 5
        # # tournament_size: 5
        # # distill_inheritance_chance: 0.33
        # # cover_chance: 0.1
        # # fitness_threshold: 1
        # 'decreased_crossover': load_logfiles([
        #     "c/logs/data1/2019-11-22.08:23:45.log",
        #     "c/logs/data1/2019-11-22.08:23:57.log",
        #     "c/logs/data1/2019-11-22.08:24:08.log",
        #     "c/logs/data1/2019-11-22.08:24:18.log",
        #     "c/logs/data1/2019-11-22.08:24:27.log",
        #     "c/logs/data1/2019-11-22.08:24:39.log",
        #     "c/logs/data1/2019-11-22.08:24:48.log",
        #     "c/logs/data1/2019-11-22.08:24:58.log",
        #     "c/logs/data1/2019-11-22.08:25:07.log",
        #     "c/logs/data1/2019-11-22.08:25:18.log",
        # ]),
        #
        # # rule_count: 60
        # # population_size: 50
        # # crossover_chance: 0.5
        # # mutation_rate: 0.003
        # # use_tournament_selection: 1
        # # selection_switch_threshold: 0.1
        # # covered_best_variations: 5
        # # tournament_size: 5
        # # distill_inheritance_chance: 0.33
        # # cover_chance: 0.1
        # # fitness_threshold: 1
        # 'half_population_size': load_logfiles([
        #     "c/logs/data1/2019-11-22.08:34:08.log",
        #     "c/logs/data1/2019-11-22.08:34:16.log",
        #     "c/logs/data1/2019-11-22.08:34:23.log",
        #     "c/logs/data1/2019-11-22.08:34:31.log",
        #     "c/logs/data1/2019-11-22.08:34:38.log",
        #     "c/logs/data1/2019-11-22.08:34:45.log",
        #     "c/logs/data1/2019-11-22.08:34:52.log",
        #     "c/logs/data1/2019-11-22.08:34:59.log",
        #     "c/logs/data1/2019-11-22.08:35:05.log",
        #     "c/logs/data1/2019-11-22.08:35:13.log",
        # ]),
        #
        # 'data2_lcs': load_logfiles([
        #     "c/logs/data2/2019-11-24.06:16:24.log",
        #     "c/logs/data2/2019-11-24.06:17:30.log",
        #     "c/logs/data2/2019-11-24.06:18:33.log",
        #     "c/logs/data2/2019-11-24.06:19:49.log",
        #     "c/logs/data2/2019-11-24.06:20:55.log",
        #     "c/logs/data2/2019-11-24.06:21:57.log",
        #     "c/logs/data2/2019-11-24.06:23:20.log",
        #     "c/logs/data2/2019-11-24.06:24:22.log",
        #     "c/logs/data2/2019-11-24.06:25:27.log",
        #     "c/logs/data2/2019-11-24.06:26:33.log",
        # ]),
        #
        # # crossover: 0.85
        # # mutation: 0.003
        # 'data2_tree': load_tree_logfiles([
        #     "logs/2019-11-26 11:22:51.444333.log",
        #     "logs/2019-11-26 11:24:54.416094.log",
        #     "logs/2019-11-26 11:26:26.850369.log",
        #     "logs/2019-11-26 11:27:51.043042.log",
        #     "logs/2019-11-26 11:30:16.528040.log",
        #     "logs/2019-11-26 11:31:22.018663.log",
        #     "logs/2019-11-26 11:32:39.541524.log",
        #     "logs/2019-11-26 11:34:16.896466.log",
        #     "logs/2019-11-26 11:35:29.016209.log",
        #     "logs/2019-11-26 11:37:26.206267.log",
        # ]),
        #
        # # crossover: 0.05
        # # mutation: 0.05
        # 'data2_tree_attempt2': load_logfiles([
        #     "logs/2019-11-26 12:48:56.313553.log",
        #     "logs/2019-11-26 12:50:18.238126.log",
        #     "logs/2019-11-26 12:52:05.559672.log",
        #     "logs/2019-11-26 12:53:41.821142.log",
        #     "logs/2019-11-26 12:55:27.117888.log",
        #     "logs/2019-11-26 12:56:50.880219.log",
        #     "logs/2019-11-26 12:58:52.824440.log",
        #     "logs/2019-11-26 13:00:40.558165.log",
        #     "logs/2019-11-26 13:02:17.178818.log",
        #     "logs/2019-11-26 13:04:12.115433.log",
        # ]),
        #
        # # crossover: 0.40
        # # mutation: 0.001
        # 'data2_tree_attempt3': load_logfiles([
        #     "logs/2019-11-26 13:37:20.011018.log",
        #     "logs/2019-11-26 13:40:06.673122.log",
        #     "logs/2019-11-26 13:42:10.223473.log",
        #     "logs/2019-11-26 13:43:48.611438.log",
        #     "logs/2019-11-26 13:45:37.417914.log",
        #     "logs/2019-11-26 13:47:45.237430.log",
        #     "logs/2019-11-26 13:50:59.602199.log",
        #     "logs/2019-11-26 13:52:08.075447.log",
        #     "logs/2019-11-26 13:53:00.380131.log",
        #     "logs/2019-11-26 13:55:26.768978.log",
        # ]),
        #
        # 'data2_tree_attempt4': load_logfiles([
        #     "logs/2019-11-26 14:09:44.676483.log",
        #     "logs/2019-11-26 14:12:34.567685.log",
        #     "logs/2019-11-26 14:14:43.562317.log",
        #     "logs/2019-11-26 14:15:54.442117.log",
        #     "logs/2019-11-26 14:18:03.070736.log",
        #     "logs/2019-11-26 14:19:59.346386.log",
        #     "logs/2019-11-26 14:21:03.909699.log",
        #     "logs/2019-11-26 14:24:10.161734.log",
        #     "logs/2019-11-26 14:26:02.237139.log",
        # ]),

        # 'data2_tree_parameter_sweeps': load_logfiles([
        #     "logs/2019-11-26 17:02:33.639160.log",
        #     "logs/2019-11-26 17:03:16.390764.log",
        #     "logs/2019-11-26 17:04:23.668007.log",
        #     "logs/2019-11-26 17:05:07.702961.log",
        #     "logs/2019-11-26 17:06:41.019835.log",
        #     "logs/2019-11-26 17:08:55.737459.log",
        #     "logs/2019-11-26 17:10:46.998311.log",
        #     "logs/2019-11-26 17:13:04.921008.log",
        #     "logs/2019-11-26 17:15:06.616866.log",
        #     "logs/2019-11-26 17:15:30.767861.log",
        #     "logs/2019-11-26 17:16:00.387150.log",
        #     "logs/2019-11-26 17:17:50.305235.log",
        #     "logs/2019-11-26 17:19:17.550643.log",
        #     "logs/2019-11-26 17:20:19.019303.log",
        #     "logs/2019-11-26 17:21:49.388252.log",
        #     "logs/2019-11-26 17:23:00.816491.log",
        #     "logs/2019-11-26 17:24:17.263110.log",
        #     "logs/2019-11-26 17:25:11.409661.log",
        #     "logs/2019-11-26 17:28:16.256429.log",
        #     "logs/2019-11-26 17:29:20.833855.log",
        #     "logs/2019-11-26 17:30:45.076912.log",
        #     "logs/2019-11-26 17:31:32.357365.log",
        #     "logs/2019-11-26 17:32:46.429155.log",
        #     "logs/2019-11-26 17:34:51.458568.log",
        #     "logs/2019-11-26 17:36:45.522567.log",
        #     "logs/2019-11-26 17:37:45.545843.log",
        #     "logs/2019-11-26 17:38:44.404423.log",
        #     "logs/2019-11-26 17:40:22.292871.log",
        #     "logs/2019-11-26 17:41:09.543010.log",
        #     "logs/2019-11-26 17:42:17.843305.log",
        #     "logs/2019-11-26 17:43:37.958013.log",
        #     "logs/2019-11-26 17:44:20.487953.log",
        #     "logs/2019-11-26 17:44:58.392204.log",
        #     "logs/2019-11-26 17:45:44.811642.log",
        #     "logs/2019-11-26 17:46:25.008513.log",
        #     "logs/2019-11-26 17:47:11.160183.log",
        #     "logs/2019-11-26 17:47:59.673371.log",
        #     "logs/2019-11-26 17:49:01.957997.log",
        #     "logs/2019-11-26 17:49:52.260326.log",
        #     "logs/2019-11-26 17:50:38.428284.log",
        #     "logs/2019-11-26 17:51:24.734752.log",
        #     "logs/2019-11-26 17:52:11.071327.log",
        #     "logs/2019-11-26 17:52:59.823267.log",
        #     "logs/2019-11-26 17:54:01.731238.log",
        #     "logs/2019-11-26 17:54:45.141648.log",
        #     "logs/2019-11-26 17:55:30.904151.log",
        #     "logs/2019-11-26 17:56:06.822259.log",
        #     "logs/2019-11-26 17:56:39.843513.log",
        #     "logs/2019-11-26 17:57:26.633853.log",
        #     "logs/2019-11-26 17:58:14.585318.log",
        #     "logs/2019-11-26 17:59:11.014583.log",
        #     "logs/2019-11-26 17:59:46.424348.log",
        #     "logs/2019-11-26 18:00:28.437415.log",
        #     "logs/2019-11-26 18:01:14.108586.log",
        #     "logs/2019-11-26 18:01:46.413854.log",
        #     "logs/2019-11-26 18:02:28.456163.log",
        #     "logs/2019-11-26 18:03:25.325417.log",
        #     "logs/2019-11-26 18:04:32.089591.log",
        #     "logs/2019-11-26 18:05:39.572202.log",
        #     "logs/2019-11-26 18:06:40.396879.log",
        #     "logs/2019-11-26 18:07:39.817169.log",
        #     "logs/2019-11-26 18:08:36.423781.log",
        #     "logs/2019-11-26 18:09:37.184695.log",
        #     "logs/2019-11-26 18:10:37.808744.log",
        #     "logs/2019-11-26 18:11:37.338236.log",
        #     "logs/2019-11-26 18:12:36.453996.log",
        #     "logs/2019-11-26 18:13:38.487145.log",
        #     "logs/2019-11-26 18:14:43.312621.log",
        #     "logs/2019-11-26 18:15:48.388251.log",
        #     "logs/2019-11-26 18:16:55.606842.log",
        #     "logs/2019-11-26 18:17:54.814067.log",
        #     "logs/2019-11-26 18:19:02.120267.log",
        #     "logs/2019-11-26 18:19:56.773425.log",
        #     "logs/2019-11-26 18:21:06.242244.log",
        #     "logs/2019-11-26 18:22:10.950215.log",
        #     "logs/2019-11-26 18:23:15.893325.log",
        #     "logs/2019-11-26 18:24:20.820685.log",
        #     "logs/2019-11-26 18:25:33.956851.log",
        #     "logs/2019-11-26 18:26:29.597788.log",
        #     "logs/2019-11-26 18:27:29.594244.log",
        # ]),

        'data2_gp_parameter_sweeps': load_logfiles([
            "logs/2019-11-27 10:42:16.088652.log",
            "logs/2019-11-27 10:44:22.607960.log",
            "logs/2019-11-27 10:45:09.208981.log",
            "logs/2019-11-27 10:45:53.139958.log",
            "logs/2019-11-27 10:47:40.578790.log",
            "logs/2019-11-27 10:50:46.531970.log",
            "logs/2019-11-27 10:51:24.152525.log",
            "logs/2019-11-27 10:52:36.660135.log",
            "logs/2019-11-27 10:53:06.762727.log",
            "logs/2019-11-27 10:53:40.524748.log",
            "logs/2019-11-27 10:54:14.987305.log",
            "logs/2019-11-27 10:55:01.130677.log",
            "logs/2019-11-27 10:56:09.696378.log",
            "logs/2019-11-27 10:57:55.250089.log",
            "logs/2019-11-27 10:58:57.428980.log",
            "logs/2019-11-27 11:01:12.759517.log",
            "logs/2019-11-27 11:02:46.375869.log",
            "logs/2019-11-27 11:03:21.999633.log",
            "logs/2019-11-27 11:06:03.313727.log",
            "logs/2019-11-27 11:09:36.308939.log",
            "logs/2019-11-27 11:10:38.099367.log",
            "logs/2019-11-27 11:12:31.772449.log",
            "logs/2019-11-27 11:13:03.790890.log",
            "logs/2019-11-27 11:14:09.309308.log",
            "logs/2019-11-27 11:15:23.554431.log",
            "logs/2019-11-27 11:16:18.868224.log",
            "logs/2019-11-27 11:18:49.691231.log",
            "logs/2019-11-27 11:20:19.307859.log",
            "logs/2019-11-27 11:22:27.445084.log",
            "logs/2019-11-27 11:25:14.005730.log",
            "logs/2019-11-27 11:27:20.649559.log",
            "logs/2019-11-27 11:28:25.636019.log",
            "logs/2019-11-27 11:28:51.205865.log",
            "logs/2019-11-27 11:31:44.627541.log",
            "logs/2019-11-27 11:34:11.330360.log",
            "logs/2019-11-27 11:34:40.045008.log",
            "logs/2019-11-27 11:35:58.110266.log",
            "logs/2019-11-27 11:36:28.923981.log",
            "logs/2019-11-27 11:37:18.216955.log",
            "logs/2019-11-27 11:38:06.418770.log",
            "logs/2019-11-27 11:39:42.488448.log",
            "logs/2019-11-27 11:41:01.545086.log",
            "logs/2019-11-27 11:43:09.770434.log",
            "logs/2019-11-27 11:44:21.558699.log",
            "logs/2019-11-27 11:45:21.474394.log",
            "logs/2019-11-27 11:47:18.685165.log",
            "logs/2019-11-27 11:48:20.324584.log",
            "logs/2019-11-27 11:49:46.822288.log",
            "logs/2019-11-27 11:51:05.234998.log",
            "logs/2019-11-27 11:52:31.284554.log",
            "logs/2019-11-27 11:53:55.437517.log",
            "logs/2019-11-27 11:55:09.937104.log",
            "logs/2019-11-27 11:56:38.828609.log",
            "logs/2019-11-27 11:57:34.613987.log",
            "logs/2019-11-27 11:58:28.783463.log",
            "logs/2019-11-27 11:59:37.319933.log",
            "logs/2019-11-27 12:00:50.596179.log",
            "logs/2019-11-27 12:01:34.899210.log",
            "logs/2019-11-27 12:02:34.630470.log",
            "logs/2019-11-27 12:03:27.537279.log",
            "logs/2019-11-27 12:04:31.648962.log",
            "logs/2019-11-27 12:05:32.104866.log",
            "logs/2019-11-27 12:06:23.878088.log",
            "logs/2019-11-27 12:07:24.112758.log",
            "logs/2019-11-27 12:08:24.559047.log",
            "logs/2019-11-27 12:09:25.360223.log",
            "logs/2019-11-27 12:10:34.739175.log",
            "logs/2019-11-27 12:11:29.152154.log",
            "logs/2019-11-27 12:12:19.929563.log",
            "logs/2019-11-27 12:13:15.156566.log",
            "logs/2019-11-27 12:14:14.279044.log",
            "logs/2019-11-27 12:15:06.285209.log",
            "logs/2019-11-27 12:16:00.394673.log",
            "logs/2019-11-27 12:16:58.877813.log",
            "logs/2019-11-27 12:17:47.425020.log",
            "logs/2019-11-27 12:18:46.228986.log",
            "logs/2019-11-27 12:20:39.987231.log",
            "logs/2019-11-27 12:22:34.413530.log",
            "logs/2019-11-27 12:24:32.419921.log",
            "logs/2019-11-27 12:26:35.473332.log",
            "logs/2019-11-27 12:28:28.559227.log",
            "logs/2019-11-27 12:30:38.674905.log",
            "logs/2019-11-27 12:32:51.003539.log",
            "logs/2019-11-27 12:35:04.583513.log",
            "logs/2019-11-27 12:37:22.287548.log",
            "logs/2019-11-27 12:39:34.618087.log",
            "logs/2019-11-27 12:41:53.656166.log",
            "logs/2019-11-27 12:44:18.566509.log",
            "logs/2019-11-27 12:46:32.982994.log",
            "logs/2019-11-27 12:49:00.019381.log",
            "logs/2019-11-27 12:51:23.340010.log",
            "logs/2019-11-27 12:53:51.986488.log",
            "logs/2019-11-27 12:56:12.269812.log",
            "logs/2019-11-27 12:58:42.505704.log",
            "logs/2019-11-27 13:01:16.334998.log",
            "logs/2019-11-27 13:03:49.626005.log",
            "logs/2019-11-27 13:06:28.744159.log",
            "logs/2019-11-27 13:09:12.212224.log",
            "logs/2019-11-27 13:11:56.769722.log",
            "logs/2019-11-27 13:13:54.366796.log",
        ]),

        'data1_parameter_sweeps': load_logfiles([
            "c/logs/data1/2019-11-27.12:25:17.log",
            "c/logs/data1/2019-11-27.12:25:44.log",
            "c/logs/data1/2019-11-27.12:26:14.log",
            "c/logs/data1/2019-11-27.12:26:53.log",
            "c/logs/data1/2019-11-27.12:27:27.log",
            "c/logs/data1/2019-11-27.12:28:08.log",
            "c/logs/data1/2019-11-27.12:28:20.log",
            "c/logs/data1/2019-11-27.12:28:34.log",
            "c/logs/data1/2019-11-27.12:28:48.log",
            "c/logs/data1/2019-11-27.12:29:03.log",
            "c/logs/data1/2019-11-27.12:29:21.log",
            "c/logs/data1/2019-11-27.12:29:29.log",
            "c/logs/data1/2019-11-27.12:29:37.log",
            "c/logs/data1/2019-11-27.12:29:48.log",
            "c/logs/data1/2019-11-27.12:30:00.log",
            "c/logs/data1/2019-11-27.12:30:21.log",
            "c/logs/data1/2019-11-27.12:31:00.log",
            "c/logs/data1/2019-11-27.12:31:43.log",
            "c/logs/data1/2019-11-27.12:32:27.log",
            "c/logs/data1/2019-11-27.12:33:15.log",
            "c/logs/data1/2019-11-27.12:34:07.log",
            "c/logs/data1/2019-11-27.12:34:36.log",
            "c/logs/data1/2019-11-27.12:35:06.log",
            "c/logs/data1/2019-11-27.12:35:44.log",
            "c/logs/data1/2019-11-27.12:36:22.log",
            "c/logs/data1/2019-11-27.12:37:05.log",
            "c/logs/data1/2019-11-27.12:37:17.log",
            "c/logs/data1/2019-11-27.12:37:29.log",
            "c/logs/data1/2019-11-27.12:37:43.log",
            "c/logs/data1/2019-11-27.12:38:01.log",
            "c/logs/data1/2019-11-27.12:38:18.log",
            "c/logs/data1/2019-11-27.12:38:25.log",
            "c/logs/data1/2019-11-27.12:38:33.log",
            "c/logs/data1/2019-11-27.12:38:47.log",
            "c/logs/data1/2019-11-27.12:39:05.log",
            "c/logs/data1/2019-11-27.12:39:26.log",
            "c/logs/data1/2019-11-27.12:40:03.log",
            "c/logs/data1/2019-11-27.12:40:44.log",
            "c/logs/data1/2019-11-27.12:41:30.log",
            "c/logs/data1/2019-11-27.12:42:19.log",
            "c/logs/data1/2019-11-27.12:43:13.log",
            "c/logs/data1/2019-11-27.12:43:43.log",
            "c/logs/data1/2019-11-27.12:44:17.log",
            "c/logs/data1/2019-11-27.12:44:53.log",
            "c/logs/data1/2019-11-27.12:45:35.log",
            "c/logs/data1/2019-11-27.12:46:16.log",
            "c/logs/data1/2019-11-27.12:46:26.log",
            "c/logs/data1/2019-11-27.12:46:40.log",
            "c/logs/data1/2019-11-27.12:46:53.log",
            "c/logs/data1/2019-11-27.12:47:10.log",
            "c/logs/data1/2019-11-27.12:47:30.log",
            "c/logs/data1/2019-11-27.12:47:38.log",
            "c/logs/data1/2019-11-27.12:47:46.log",
            "c/logs/data1/2019-11-27.12:47:54.log",
            "c/logs/data1/2019-11-27.12:48:10.log",
            "c/logs/data1/2019-11-27.12:48:31.log",
            "c/logs/data1/2019-11-27.12:49:09.log",
            "c/logs/data1/2019-11-27.12:49:51.log",
            "c/logs/data1/2019-11-27.12:50:38.log",
            "c/logs/data1/2019-11-27.12:51:28.log",
            "c/logs/data1/2019-11-27.12:52:23.log",
            "c/logs/data1/2019-11-27.12:52:54.log",
            "c/logs/data1/2019-11-27.12:53:29.log",
            "c/logs/data1/2019-11-27.12:54:09.log",
            "c/logs/data1/2019-11-27.12:54:51.log",
            "c/logs/data1/2019-11-27.12:55:37.log",
            "c/logs/data1/2019-11-27.12:55:50.log",
            "c/logs/data1/2019-11-27.12:56:02.log",
            "c/logs/data1/2019-11-27.12:56:20.log",
            "c/logs/data1/2019-11-27.12:56:39.log",
            "c/logs/data1/2019-11-27.12:56:55.log",
            "c/logs/data1/2019-11-27.12:57:03.log",
            "c/logs/data1/2019-11-27.12:57:14.log",
            "c/logs/data1/2019-11-27.12:57:27.log",
            "c/logs/data1/2019-11-27.12:57:36.log",
            "c/logs/data1/2019-11-27.12:58:01.log",
            "c/logs/data1/2019-11-27.12:58:40.log",
            "c/logs/data1/2019-11-27.12:59:14.log",
            "c/logs/data1/2019-11-27.13:00:04.log",
            "c/logs/data1/2019-11-27.13:00:57.log",
            "c/logs/data1/2019-11-27.13:01:56.log",
            "c/logs/data1/2019-11-27.13:02:23.log",
            "c/logs/data1/2019-11-27.13:02:59.log",
            "c/logs/data1/2019-11-27.13:03:44.log",
            "c/logs/data1/2019-11-27.13:04:26.log",
            "c/logs/data1/2019-11-27.13:05:15.log",
            "c/logs/data1/2019-11-27.13:05:28.log",
            "c/logs/data1/2019-11-27.13:05:42.log",
            "c/logs/data1/2019-11-27.13:05:55.log",
            "c/logs/data1/2019-11-27.13:06:10.log",
            "c/logs/data1/2019-11-27.13:06:31.log",
            "c/logs/data1/2019-11-27.13:06:38.log",
            "c/logs/data1/2019-11-27.13:06:49.log",
            "c/logs/data1/2019-11-27.13:06:59.log",
            "c/logs/data1/2019-11-27.13:07:20.log",
            "c/logs/data1/2019-11-27.13:07:33.log",
            "c/logs/data1/2019-11-27.13:08:14.log",
            "c/logs/data1/2019-11-27.13:08:59.log",
            "c/logs/data1/2019-11-27.13:09:50.log",
            "c/logs/data1/2019-11-27.13:10:45.log",
        ])
    }

    # for name, description in {
    #     'dynamic_mutation': 'data1 - mutation adjusted by rule size',
    #     'static_mutation': 'data1 - fixed mutation rate (0.003)',
    #     'roulette_wheel_selection': 'data1 - roulette wheel selection',
    #     'boosted_mutation': 'data1 - boosted mutation rate (0.01)',
    #     'decreased_crossover': 'data1 - decreased crossover rate 85% -> 50%',
    #     'half_population_size': 'data1 - half population size 100 -> 50',
    # }.items():
    #     for key in ['train_fitness_best', 'train_fitness_mean']:
    #         plot_fitness_area(500, experiments[name], key)
    #     plt.xlabel("generations")
    #     plt.ylabel("fitness")
    #     plt.legend(loc="lower right")
    #     plt.title(description)
    #     plt.savefig(f'graphs/{name}.png')
    #     plt.show()
    #
    # for name, description in {'data2_lcs': 'data2 - model by rules'}.items():
    #     for key in ['train_fitness_best', 'train_fitness_mean']:
    #         plot_fitness_area(1000, experiments[name], key)
    #     plt.xlabel("generations")
    #     plt.ylabel("fitness")
    #     plt.legend(loc="upper left")
    #     plt.title(description)
    #     plt.savefig(f'graphs/{name}.png')
    #     plt.show()

    # for name, description in {'data2_tree': 'data2 - model by tree'}.items():
    #     for key in ['Best', 'Mean']:
    #         plot_fitness_area(1000, experiments[name], key)
    #     plt.xlabel("generations")
    #     plt.ylabel("fitness")
    #     plt.legend(loc="upper left")
    #     plt.title(description)
    #     plt.savefig(f'graphs/{name}.png')
    #     plt.show()

    # for name, description in {
    #     'data2_tree_attempt2': 'data2 - genetic programmer (mutation 0.05, crossover 0.05)',
    #     'data2_tree_attempt3': 'data2 - genetic programmer (mutation 0.001, crossover 0.4)',
    #     'data2_tree_attempt4': 'data2 - genetic programmer (mutation 0.001, crossover 0.85)',
    # }.items():
    #     for key in ['train_fitness_best', 'train_fitness_mean']:
    #         plot_fitness_area(1000, experiments[name], key)
    #     plt.xlabel("generations")
    #     plt.ylabel("fitness")
    #     plt.legend(loc="lower left")
    #     plt.title(description)
    #     plt.savefig(f'graphs/{name}.png')
    #     plt.show()

    # for name, description in {
    #     'data2_tree_attempt2': 'data2 - genetic programmer (mutation 0.05, crossover 0.05)',
    #     'data2_tree_attempt3': 'data2 - genetic programmer (mutation 0.001, crossover 0.4)',
    # }.items():
    #     for key in ['train_fitness_best', 'train_fitness_mean']:
    #         plot_fitness_area(1000, experiments[name], key)
    #     plt.xlabel("generations")
    #     plt.ylabel("fitness")
    #     plt.legend(loc="lower left")
    #     plt.title(description)
    #     plt.savefig(f'graphs/{name}.png')
    #     plt.show()

    plot_parameter_sweep('data1 - LCS parameter sweep',
                         experiments['data1_parameter_sweeps'])
    plot_parameter_sweep('data2 - genetic programmer parameter sweep', experiments['data2_gp_parameter_sweeps'])


if __name__ == '__main__':
    main()
