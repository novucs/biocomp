solution = "0,1,#,0,#,#,0,1,0,#,#,1,#,1,#,1,#,#,#,1,1,0,1,#,#,#,#,1,0,#,1,#,#,#,1,#,#,#,#,#,#,0"
rule_size = 7
solution = solution.split(',')

for i, j in enumerate(range(0, len(solution), rule_size)):
    print(f'Rule #{i}:\t{" ".join(solution[j:j + rule_size])}')
