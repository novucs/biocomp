# solution = "#,0,0,0,1,#,1,0,#,#,#,1,0,0,1,1,1,1,#,#,1,0,0,1,1,#,#,0,1,0,1,0,0,#,0,1,0,0,1,0,#,1,#,#,#,#,0,1,0,0,1,1,0,#,#,0,#,#,1,1,#,1,1,0,1,0,1,#,#,0,0,0,0,0,#,#,0,#,#,0,0,1,0,1,1,0,#,#,#,1,0,1,1,0,0,#,#,0,#,#,1,1,1,#,0,#,1,#,#,1,0,0,#,#,#,#,#,#,1"
solution = "0,1,0,1,0,1,1,0,1,1,0,1,1,0,0,1,1,0,#,#,1,1,0,1,#,0,1,0,#,0,#,0,0,1,0,1,#,1,1,#,1,1,#,0,1,#,0,1,1,0,0,0,0,0,#,0,1,0,0,0,1,0,0,#,0,0,0,#,#,1,1,0,0,#,0,0,0,1,0,#,1,0,#,1,0,1,0,1,1,0,1,0,#,#,#,1,0,0,1,#,1,1,0,#,0,#,0,1,1,#,#,0,1,0,#,#,0,#,0,1,#,0,0,0,1,1,#,#,#,#,0,1,0,#,#,#,#,0,1,0,0,#,#,0,#,#,1,0,1,0,#,#,#,0,0,#,#,#,#,#,1,#,#,0,0,1,0,1,#,0,#,#,#,1,0,#,#,0,0,#,#,0,#,1,0,#,1,0,0,#,1,#,0,1,0,0,#,#,#,#,#,#,1"
rule_size = 7
solution = [int(s) if s != '#' else '#' for s in solution.split(',')]


def print_solution():
    print('Solution:')
    for i, j in enumerate(range(0, len(solution), rule_size)):
        pretty_rule = " ".join(map(str, solution[j:j + rule_size]))
        print(f'\tRule #{i}:\t{pretty_rule}')


def evaluate(chromosome, attributes):
    # votes = [0, 0]
    for index in range(0, len(solution), rule_size):
        # print(chromosome)
        *condition, action = chromosome[index: index + rule_size]
        if all(p == f or p == "#" for p, f in zip(condition, attributes)):
            # votes[action] += 1
            return action
    # return votes.index(max(votes))
    return None


def check_against_data2():
    print('Missing Values:')
    for attributes in [
        [0, 0, 0, 0, 0, 1],
        [0, 0, 1, 1, 1, 0],
        [1, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1],
    ]:
        prediction = evaluate(solution, attributes)
        print('\tAttributes:', ' '.join(map(str, attributes)), 'Predicted Label:', prediction)

    from biocomp import datasets
    train_x, train_y, *_ = datasets.split(datasets.load_dataset_2())
    correct = 0
    for features, label in zip(train_x, train_y):
        if evaluate(solution, features) == label:
            correct += 1
    fitness = correct / len(train_x)
    print()
    print(f'Achieved {correct}/{len(train_x)} ({fitness}) on training data')


def print_hard_coded_rules():
    rules = []
    for i in range(0, len(solution), rule_size):
        rule = solution[i:i + rule_size]
        if '#' not in rule:
            rules.append(rule)

    print(f'Found {len(rules)} hardcoded rules:')
    for rule in rules:
        print(f'\t{"".join(map(str, rule[:-1]))} {rule[-1]}')


def main():
    print_solution()
    print()
    check_against_data2()
    print()
    print_hard_coded_rules()


if __name__ == '__main__':
    main()
