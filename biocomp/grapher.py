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


def plot_logfile(logs, keys):
    generation_count = 300
    for key in keys:
        plot_fitness_area(generation_count, logs, key)
    plt.xlabel("generations")
    plt.ylabel("fitness")
    plt.legend(loc="upper left")


def main():
    experiments = [
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
        load_logfiles([
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

            "c/logs/data1/2019-11-22.06:49:03.log",
            "c/logs/data1/2019-11-22.06:49:13.log",
            "c/logs/data1/2019-11-22.06:49:25.log",
            "c/logs/data1/2019-11-22.06:49:36.log",
            "c/logs/data1/2019-11-22.06:49:46.log",
            "c/logs/data1/2019-11-22.06:50:00.log",
            "c/logs/data1/2019-11-22.06:50:11.log",
            "c/logs/data1/2019-11-22.06:50:23.log",
            "c/logs/data1/2019-11-22.06:50:35.log",
            "c/logs/data1/2019-11-22.06:50:46.log",
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
        load_logfiles([
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
        load_logfiles([
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
    ]

    plot_logfile(experiments[0], ['train_fitness_best', 'train_fitness_mean'])
    plt.title('data1 - mutation adjusted by rule size')
    plt.show()
    plot_logfile(experiments[1], ['train_fitness_best', 'train_fitness_mean'])
    plt.title('data1 - fixed mutation rate')
    plt.show()


if __name__ == '__main__':
    main()
