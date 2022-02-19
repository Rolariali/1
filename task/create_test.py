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
    with open(f'task/{file_name}.h', 'w') as f:
        f.write(input)


d = Data()

d.add(96)

for _ in range(round(qty/3)):
    # s += f'{random.randint(1,100)},'
    d.add(97)
d.add(1)
for _ in range(round(qty/3)):
    # s += f'{random.randint(1,100)},'
    d.add(98)

for _ in range(round(qty/3)):
    # s += f'{random.randint(1,100)},'
    d.add(99)

write_h("pb96-1", d.s)
# s += f'{random.randint(1,100)}'


print('done!')

