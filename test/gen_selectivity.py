width = 32
code_width = "width_{}".format(width)

# compile file

max_val = 1 << width

targets = ""

operator = "bt"

for selectivity in range(0,110,10):
    target = int((max_val - 1) * selectivity / 100)
    targets += str(target)
    targets += ","
    for col_num in range(0,3):
        print("{} {} {}".format(operator, 0, target))

print()
targets = ""
operator = "lt"
for selectivity in range(0,110,10):
    target = int((max_val - 1) * selectivity / 100)
    targets += str(target)
    targets += ","
    for col_num in range(0,3):
        print("{} {}".format(operator, target))

print()
targets = ""
operator1 = "lt"
operator2 = "eq"
for selectivity in range(0,110,10):
    target = int((max_val - 1) * selectivity / 100)
    targets += str(target)
    targets += ","
    for col_num in range(0,3):
        if col_num == 0:
            print("{} {}".format(operator2, target))
        else:
            print("{} {}".format(operator1, target))

print()
targets = ""
operator = "eq"
for selectivity in range(0,110,10):
    target = int((max_val - 1) * selectivity / 100)
    targets += str(target)
    targets += ","
    for col_num in range(0,3):
        print("{} {}".format(operator, target))