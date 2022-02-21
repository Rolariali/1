import random
import sys


qty = int(sys.argv[1])

class Data:

    s = ""

    def add(self, i: int):
        self.s += f"{i},"


def _set_vector(s: str) -> str:
    input = "std::vector<uint8_t> input = { " +  s  + " };"
    return input

def write_h(file_name: str, s: str):
    input = _set_vector(s)
    with open(f'task/data/{file_name}.h', 'w') as f:
        f.write(input)


d = Data()

for _ in range(14):
    d.add(99)

for _ in range(1111):
    d.add(101)

for _ in range(1111):
    d.add(103)

for _ in range(1111):
    d.add(105)

for _ in range(11111):
    d.add(106)

for _ in range(111111):
    d.add(106)

for _ in range(1111111):
    d.add(108)

write_h("big_delta", d.s)
# s += f'{random.randint(1,100)}'


print('done!')

