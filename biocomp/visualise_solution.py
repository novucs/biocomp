from biocomp import datasets

# solution = "#,0,0,0,1,#,1,0,#,#,#,1,0,0,1,1,1,1,#,#,1,0,0,1,1,#,#,0,1,0,1,0,0,#,0,1,0,0,1,0,#,1,#,#,#,#,0,1,0,0,1,1,0,#,#,0,#,#,1,1,#,1,1,0,1,0,1,#,#,0,0,0,0,0,#,#,0,#,#,0,0,1,0,1,1,0,#,#,#,1,0,1,1,0,0,#,#,0,#,#,1,1,1,#,0,#,1,#,#,1,0,0,#,#,#,#,#,#,1"
# solution = "1,1,0,1,1,0,0,0,1,1,1,0,1,0,0,0,#,1,1,0,0,0,1,1,0,1,0,1,0,0,1,0,0,0,1,0,1,0,1,0,0,0,#,1,1,1,#,1,1,1,1,1,0,0,0,1,#,0,1,0,0,#,0,0,#,1,#,0,1,1,1,1,0,0,1,1,0,1,0,0,1,0,0,0,#,#,1,#,0,1,0,0,0,#,0,#,1,1,1,1,#,0,#,1,1,#,1,1,0,#,#,0,#,#,0,1,#,0,1,1,1,#,#,1,#,1,0,0,#,#,0,#,0,0,#,1,#,1,#,0,#,#,0,0,1,1,1,#,#,#,1,1,0,0,#,#,#,0,#,1,0,0,#,#,#,0,#,1,#,1,#,#,#,#,0,#,#,#,#,0,#,1,1,#,0,#,#,#,0,#,#,#,#,#,#,1"
# rule_size = 7
# solution = [int(float(s)) if s != '#' else '#' for s in solution.split(',')]

# solution learned from digital data
"""
##01#####0 1
#####0###0 0
##1####0#1 0
##1####### 1
######1##1 1
########## 0
"""

# solution learned from floating data
"""
##01#####0 1
##1##1###0 1
##1####0#1 0
##1######1 1
######1##1 1
########## 0
"""

rule_size = 11
solution = """
##01#####0 1
##0###1##1 1
##1##1###0 1
##1####1#1 1
########## 0
""".strip().replace(" ", "").replace("\n", "")
solution = [int(float(s)) if s != '#' else '#' for s in solution]


def print_solution():
    print('Solution:')
    for i, j in enumerate(range(0, len(solution), rule_size)):
        rule = solution[j:j + rule_size]
        pretty_rule = "".join(map(str, rule[:-1])) + " " + str(rule[-1])
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
    print('Missing Value Predictions:')
    for attributes in [
        [0, 0, 0, 0, 0, 1],
        [0, 0, 1, 1, 1, 0],
        [1, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1],
    ]:
        prediction = evaluate(solution, attributes)
        print('\tAttributes:', ''.join(map(str, attributes)), 'Predicted Label:', prediction)

    train_x, train_y, *_ = datasets.split(datasets.load_dataset_2())
    correct = 0
    for features, label in zip(train_x, train_y):
        if evaluate(solution, features) == label:
            correct += 1
        else:
            print("\tUnable to correctly predict:", ''.join(map(str, map(int, features))), int(label))
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
        rule_string = "".join(map(str, rule[:-1]))
        print(f'\t{rule_string} {rule[-1]} {int(rule_string, 2)}')


def check_against_digital_data4():
    X, y, *_ = datasets.split(datasets.load_dataset(
        "../datasets/2019/digital_data4.txt", datasets.parse_binary_string_features))

    failed_predictions = []

    for features, label in zip(X, y):
        for i, j in enumerate(range(0, len(solution), rule_size)):
            *condition, prediction = solution[j:j + rule_size]

            if ''.join(map(str, map(int, features))) == '0000001011' and ''.join(map(str, condition)) == '######1##1':
                print("")

            if all(c == f or c == '#' for c, f in zip(condition, features)):
                if prediction != label:
                    failed_predictions.append((features, condition, label))
                break

    print(f"Classified {len(X) - len(failed_predictions)} / {len(X)}")

    for features, condition, label in failed_predictions:
        print(''.join(map(str, condition)), "did not predict",
              ''.join(map(str, map(int, features))), int(label))



def check_data2_tree():
    solution = '(((f1+f4)-((f5+f0)-(f2-(1+f3))))%2)<0.1'
    tx, ty = datasets.split(datasets.load_dataset_2())
    correct_count = 0

    for features, labels in zip(tx, ty):
        prediction = eval(solution, {f'f{i}': v for i, v in enumerate(features)})
        if prediction == labels:
            correct_count += 1
        else:
            print('Could not predict: ', ''.join(map(str, map(int, features))))

    print(f'Predicted {correct_count}/{len(tx)} correct for data2')


def main():
    print_solution()
    # print()
    check_against_data2()
    # print()
    # print_hard_coded_rules()
    # check_against_digital_data4()
    # check_data2_tree()


if __name__ == '__main__':
    main()
