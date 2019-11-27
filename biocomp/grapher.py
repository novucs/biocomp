import re

import matplotlib.pyplot as plt


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
        entries = [dict(v.split(":", maxsplit=1) for v in l.split(" ")) for l in lines if "generation" in l]

        if 'rule_count' not in settings:
            settings['rule_count'] = 1

        for entry in entries:
            if 'rule_count' not in entry:
                entry['rule_count'] = 0

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
            # fitness = float(entry[key])
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


def main():
    experiments = {
        # rule_count:60
        # population_size: 100
        # crossover_chance: 0.85
        # mutation_rate: 1
        # selection_switch_threshold: 0.1
        # covered_best_variations: 5
        # tournament_size: 5
        # distill_inheritance_chance: 0.33
        # cover_chance: 0.1
        # fitness_threshold: 1
        'dynamic_mutation': load_logfiles([
            "c/logs/data1/2019-11-22.05:30:55.log",
            "c/logs/data1/2019-11-22.05:31:31.log",
            "c/logs/data1/2019-11-22.05:32:06.log",
            "c/logs/data1/2019-11-22.05:32:30.log",
            "c/logs/data1/2019-11-22.05:35:27.log",
            "c/logs/data1/2019-11-22.05:36:01.log",
            "c/logs/data1/2019-11-22.05:36:49.log",
            "c/logs/data1/2019-11-22.05:37:32.log",
            "c/logs/data1/2019-11-22.05:38:35.log",
            "c/logs/data1/2019-11-22.05:39:12.log",

            # "c/logs/data1/2019-11-22.06:49:03.log",
            # "c/logs/data1/2019-11-22.06:49:13.log",
            # "c/logs/data1/2019-11-22.06:49:25.log",
            # "c/logs/data1/2019-11-22.06:49:36.log",
            # "c/logs/data1/2019-11-22.06:49:46.log",
            # "c/logs/data1/2019-11-22.06:50:00.log",
            # "c/logs/data1/2019-11-22.06:50:11.log",
            # "c/logs/data1/2019-11-22.06:50:23.log",
            # "c/logs/data1/2019-11-22.06:50:35.log",
            # "c/logs/data1/2019-11-22.06:50:46.log",
        ]),

        # rule_count:60
        # population_size: 100
        # crossover_chance: 0.85
        # mutation_rate: 0.003
        # selection_switch_threshold: 0.1
        # covered_best_variations: 5
        # tournament_size: 5
        # distill_inheritance_chance: 0.33
        # cover_chance: 0.1
        # fitness_threshold: 1
        'static_mutation': load_logfiles([
            "c/logs/data1/2019-11-22.07:07:02.log",
            "c/logs/data1/2019-11-22.07:07:14.log",
            "c/logs/data1/2019-11-22.07:07:28.log",
            "c/logs/data1/2019-11-22.07:07:40.log",
            "c/logs/data1/2019-11-22.07:07:52.log",
            "c/logs/data1/2019-11-22.07:08:03.log",
            "c/logs/data1/2019-11-22.07:08:17.log",
            "c/logs/data1/2019-11-22.07:08:30.log",
            "c/logs/data1/2019-11-22.07:08:42.log",
            "c/logs/data1/2019-11-22.07:08:55.log",

            # "c/logs/data1/2019-11-22.07:46:00.log",
            # "c/logs/data1/2019-11-22.07:46:11.log",
            # "c/logs/data1/2019-11-22.07:46:23.log",
            # "c/logs/data1/2019-11-22.07:46:36.log",
            # "c/logs/data1/2019-11-22.07:46:49.log",
            # "c/logs/data1/2019-11-22.07:47:01.log",
            # "c/logs/data1/2019-11-22.07:47:12.log",
            # "c/logs/data1/2019-11-22.07:47:25.log",
            # "c/logs/data1/2019-11-22.07:47:37.log",
            # "c/logs/data1/2019-11-22.07:47:48.log",
        ]),

        # rule_count:5
        # population_size: 100
        # crossover_chance: 0.85
        # mutation_rate: 0.003
        # selection_switch_threshold: 0.1
        # covered_best_variations: 5
        # tournament_size: 5
        # distill_inheritance_chance: 0.33
        # cover_chance: 0.1
        # fitness_threshold: 1.66667
        'static_rule_count': load_logfiles([
            "c/logs/data1/2019-11-22.07:17:15.log",
            "c/logs/data1/2019-11-22.07:17:22.log",
            "c/logs/data1/2019-11-22.07:17:29.log",
            "c/logs/data1/2019-11-22.07:17:36.log",
            "c/logs/data1/2019-11-22.07:17:43.log",
            "c/logs/data1/2019-11-22.07:17:50.log",
            "c/logs/data1/2019-11-22.07:17:57.log",
            "c/logs/data1/2019-11-22.07:18:04.log",
            "c/logs/data1/2019-11-22.07:18:11.log",
            "c/logs/data1/2019-11-22.07:18:18.log",
        ]),

        # rule_count:60
        # population_size: 100
        # crossover_chance: 0.85
        # mutation_rate: 0.003
        # use_tournament_selection: false
        # selection_switch_threshold: 0.1
        # covered_best_variations: 5
        # tournament_size: 5
        # distill_inheritance_chance: 0.33
        # cover_chance: 0.1
        # fitness_threshold: 1
        'roulette_wheel_selection': load_logfiles([
            "c/logs/data1/2019-11-22.08:05:56.log",
            "c/logs/data1/2019-11-22.08:06:14.log",
            "c/logs/data1/2019-11-22.08:06:35.log",
            "c/logs/data1/2019-11-22.08:06:59.log",
            "c/logs/data1/2019-11-22.08:07:20.log",
            "c/logs/data1/2019-11-22.08:07:43.log",
            "c/logs/data1/2019-11-22.08:08:05.log",
            "c/logs/data1/2019-11-22.08:08:27.log",
            "c/logs/data1/2019-11-22.08:08:51.log",
            "c/logs/data1/2019-11-22.08:09:13.log",
        ]),

        # rule_count: 60
        # population_size: 100
        # crossover_chance: 0.85
        # mutation_rate: 0.01
        # use_tournament_selection: 1
        # selection_switch_threshold: 0.1
        # covered_best_variations: 5
        # tournament_size: 5
        # distill_inheritance_chance: 0.33
        # cover_chance: 0.1
        # fitness_threshold: 1
        'boosted_mutation': load_logfiles([
            "c/logs/data1/2019-11-22.08:18:21.log",
            "c/logs/data1/2019-11-22.08:18:36.log",
            "c/logs/data1/2019-11-22.08:18:47.log",
            "c/logs/data1/2019-11-22.08:19:05.log",
            "c/logs/data1/2019-11-22.08:19:22.log",
            "c/logs/data1/2019-11-22.08:19:34.log",
            "c/logs/data1/2019-11-22.08:19:47.log",
            "c/logs/data1/2019-11-22.08:20:02.log",
            "c/logs/data1/2019-11-22.08:20:18.log",
            "c/logs/data1/2019-11-22.08:20:37.log",
        ]),

        # rule_count: 60
        # population_size: 100
        # crossover_chance: 0.5
        # mutation_rate: 0.003
        # use_tournament_selection: 1
        # selection_switch_threshold: 0.1
        # covered_best_variations: 5
        # tournament_size: 5
        # distill_inheritance_chance: 0.33
        # cover_chance: 0.1
        # fitness_threshold: 1
        'decreased_crossover': load_logfiles([
            "c/logs/data1/2019-11-22.08:23:45.log",
            "c/logs/data1/2019-11-22.08:23:57.log",
            "c/logs/data1/2019-11-22.08:24:08.log",
            "c/logs/data1/2019-11-22.08:24:18.log",
            "c/logs/data1/2019-11-22.08:24:27.log",
            "c/logs/data1/2019-11-22.08:24:39.log",
            "c/logs/data1/2019-11-22.08:24:48.log",
            "c/logs/data1/2019-11-22.08:24:58.log",
            "c/logs/data1/2019-11-22.08:25:07.log",
            "c/logs/data1/2019-11-22.08:25:18.log",
        ]),

        # rule_count: 60
        # population_size: 50
        # crossover_chance: 0.5
        # mutation_rate: 0.003
        # use_tournament_selection: 1
        # selection_switch_threshold: 0.1
        # covered_best_variations: 5
        # tournament_size: 5
        # distill_inheritance_chance: 0.33
        # cover_chance: 0.1
        # fitness_threshold: 1
        'half_population_size': load_logfiles([
            "c/logs/data1/2019-11-22.08:34:08.log",
            "c/logs/data1/2019-11-22.08:34:16.log",
            "c/logs/data1/2019-11-22.08:34:23.log",
            "c/logs/data1/2019-11-22.08:34:31.log",
            "c/logs/data1/2019-11-22.08:34:38.log",
            "c/logs/data1/2019-11-22.08:34:45.log",
            "c/logs/data1/2019-11-22.08:34:52.log",
            "c/logs/data1/2019-11-22.08:34:59.log",
            "c/logs/data1/2019-11-22.08:35:05.log",
            "c/logs/data1/2019-11-22.08:35:13.log",
        ]),

        'data2_lcs': load_logfiles([
            "c/logs/data2/2019-11-24.06:16:24.log",
            "c/logs/data2/2019-11-24.06:17:30.log",
            "c/logs/data2/2019-11-24.06:18:33.log",
            "c/logs/data2/2019-11-24.06:19:49.log",
            "c/logs/data2/2019-11-24.06:20:55.log",
            "c/logs/data2/2019-11-24.06:21:57.log",
            "c/logs/data2/2019-11-24.06:23:20.log",
            "c/logs/data2/2019-11-24.06:24:22.log",
            "c/logs/data2/2019-11-24.06:25:27.log",
            "c/logs/data2/2019-11-24.06:26:33.log",
        ]),

        # crossover: 0.85
        # mutation: 0.003
        'data2_tree': load_tree_logfiles([
            "logs/2019-11-26 11:22:51.444333.log",
            "logs/2019-11-26 11:24:54.416094.log",
            "logs/2019-11-26 11:26:26.850369.log",
            "logs/2019-11-26 11:27:51.043042.log",
            "logs/2019-11-26 11:30:16.528040.log",
            "logs/2019-11-26 11:31:22.018663.log",
            "logs/2019-11-26 11:32:39.541524.log",
            "logs/2019-11-26 11:34:16.896466.log",
            "logs/2019-11-26 11:35:29.016209.log",
            "logs/2019-11-26 11:37:26.206267.log",
        ]),

        # crossover: 0.05
        # mutation: 0.05
        'data2_tree_attempt2': load_logfiles([
            "logs/2019-11-26 12:48:56.313553.log",
            "logs/2019-11-26 12:50:18.238126.log",
            "logs/2019-11-26 12:52:05.559672.log",
            "logs/2019-11-26 12:53:41.821142.log",
            "logs/2019-11-26 12:55:27.117888.log",
            "logs/2019-11-26 12:56:50.880219.log",
            "logs/2019-11-26 12:58:52.824440.log",
            "logs/2019-11-26 13:00:40.558165.log",
            "logs/2019-11-26 13:02:17.178818.log",
            "logs/2019-11-26 13:04:12.115433.log",
        ]),

        # crossover: 0.40
        # mutation: 0.001
        'data2_tree_attempt3': load_logfiles([
            "logs/2019-11-26 13:37:20.011018.log",
            "logs/2019-11-26 13:40:06.673122.log",
            "logs/2019-11-26 13:42:10.223473.log",
            "logs/2019-11-26 13:43:48.611438.log",
            "logs/2019-11-26 13:45:37.417914.log",
            "logs/2019-11-26 13:47:45.237430.log",
            "logs/2019-11-26 13:50:59.602199.log",
            "logs/2019-11-26 13:52:08.075447.log",
            "logs/2019-11-26 13:53:00.380131.log",
            "logs/2019-11-26 13:55:26.768978.log",
        ]),

        'data2_tree_attempt4': load_logfiles([
            "logs/2019-11-26 14:09:44.676483.log",
            "logs/2019-11-26 14:12:34.567685.log",
            "logs/2019-11-26 14:14:43.562317.log",
            "logs/2019-11-26 14:15:54.442117.log",
            "logs/2019-11-26 14:18:03.070736.log",
            "logs/2019-11-26 14:19:59.346386.log",
            "logs/2019-11-26 14:21:03.909699.log",
            "logs/2019-11-26 14:24:10.161734.log",
            "logs/2019-11-26 14:26:02.237139.log",
        ]),

        'data2_tree_exhaustive': load_logfiles([
            "logs/2019-11-26 17:02:33.639160.log",
            "logs/2019-11-26 17:03:16.390764.log",
            "logs/2019-11-26 17:04:23.668007.log",
            "logs/2019-11-26 17:05:07.702961.log",
            "logs/2019-11-26 17:06:41.019835.log",
            "logs/2019-11-26 17:08:55.737459.log",
            "logs/2019-11-26 17:10:46.998311.log",
            "logs/2019-11-26 17:13:04.921008.log",
            "logs/2019-11-26 17:15:06.616866.log",
            "logs/2019-11-26 17:15:30.767861.log",
            "logs/2019-11-26 17:16:00.387150.log",
            "logs/2019-11-26 17:17:50.305235.log",
            "logs/2019-11-26 17:19:17.550643.log",
            "logs/2019-11-26 17:20:19.019303.log",
            "logs/2019-11-26 17:21:49.388252.log",
            "logs/2019-11-26 17:23:00.816491.log",
            "logs/2019-11-26 17:24:17.263110.log",
            "logs/2019-11-26 17:25:11.409661.log",
            "logs/2019-11-26 17:28:16.256429.log",
            "logs/2019-11-26 17:29:20.833855.log",
            "logs/2019-11-26 17:30:45.076912.log",
            "logs/2019-11-26 17:31:32.357365.log",
            "logs/2019-11-26 17:32:46.429155.log",
            "logs/2019-11-26 17:34:51.458568.log",
            "logs/2019-11-26 17:36:45.522567.log",
            "logs/2019-11-26 17:37:45.545843.log",
            "logs/2019-11-26 17:38:44.404423.log",
            "logs/2019-11-26 17:40:22.292871.log",
            "logs/2019-11-26 17:41:09.543010.log",
            "logs/2019-11-26 17:42:17.843305.log",
            "logs/2019-11-26 17:43:37.958013.log",
            "logs/2019-11-26 17:44:20.487953.log",
            "logs/2019-11-26 17:44:58.392204.log",
            "logs/2019-11-26 17:45:44.811642.log",
            "logs/2019-11-26 17:46:25.008513.log",
            "logs/2019-11-26 17:47:11.160183.log",
            "logs/2019-11-26 17:47:59.673371.log",
            "logs/2019-11-26 17:49:01.957997.log",
            "logs/2019-11-26 17:49:52.260326.log",
            "logs/2019-11-26 17:50:38.428284.log",
            "logs/2019-11-26 17:51:24.734752.log",
            "logs/2019-11-26 17:52:11.071327.log",
            "logs/2019-11-26 17:52:59.823267.log",
            "logs/2019-11-26 17:54:01.731238.log",
            "logs/2019-11-26 17:54:45.141648.log",
            "logs/2019-11-26 17:55:30.904151.log",
            "logs/2019-11-26 17:56:06.822259.log",
            "logs/2019-11-26 17:56:39.843513.log",
            "logs/2019-11-26 17:57:26.633853.log",
            "logs/2019-11-26 17:58:14.585318.log",
            "logs/2019-11-26 17:59:11.014583.log",
            "logs/2019-11-26 17:59:46.424348.log",
            "logs/2019-11-26 18:00:28.437415.log",
            "logs/2019-11-26 18:01:14.108586.log",
            "logs/2019-11-26 18:01:46.413854.log",
            "logs/2019-11-26 18:02:28.456163.log",
            "logs/2019-11-26 18:03:25.325417.log",
            "logs/2019-11-26 18:04:32.089591.log",
            "logs/2019-11-26 18:05:39.572202.log",
            "logs/2019-11-26 18:06:40.396879.log",
            "logs/2019-11-26 18:07:39.817169.log",
            "logs/2019-11-26 18:08:36.423781.log",
            "logs/2019-11-26 18:09:37.184695.log",
            "logs/2019-11-26 18:10:37.808744.log",
            "logs/2019-11-26 18:11:37.338236.log",
            "logs/2019-11-26 18:12:36.453996.log",
            "logs/2019-11-26 18:13:38.487145.log",
            "logs/2019-11-26 18:14:43.312621.log",
            "logs/2019-11-26 18:15:48.388251.log",
            "logs/2019-11-26 18:16:55.606842.log",
            "logs/2019-11-26 18:17:54.814067.log",
            "logs/2019-11-26 18:19:02.120267.log",
            "logs/2019-11-26 18:19:56.773425.log",
            "logs/2019-11-26 18:21:06.242244.log",
            "logs/2019-11-26 18:22:10.950215.log",
            "logs/2019-11-26 18:23:15.893325.log",
            "logs/2019-11-26 18:24:20.820685.log",
            "logs/2019-11-26 18:25:33.956851.log",
            "logs/2019-11-26 18:26:29.597788.log",
            "logs/2019-11-26 18:27:29.594244.log",
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

    for name, description in {
        'data2_tree_attempt2': 'data2 - genetic programmer (mutation 0.05, crossover 0.05)',
        'data2_tree_attempt3': 'data2 - genetic programmer (mutation 0.001, crossover 0.4)',
        'data2_tree_attempt4': 'data2 - genetic programmer (mutation 0.001, crossover 0.85)',
    }.items():
        for key in ['train_fitness_best', 'train_fitness_mean']:
            plot_fitness_area(1000, experiments[name], key)
        plt.xlabel("generations")
        plt.ylabel("fitness")
        plt.legend(loc="lower left")
        plt.title(description)
        plt.savefig(f'graphs/{name}.png')
        plt.show()


if __name__ == '__main__':
    main()
