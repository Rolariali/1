import random
import sys


qty = int(sys.argv[1])

class Data:

    s = ""

    def add(self, s: int):
        self.s += str(s)


def _set_vector(s: str) -> str:
    input = "std::vector<uint8_t> input = { " +  s  + " };"
    return input

def write_h(file_name: str, s: str):
    input = _set_vector(s)
    with open(f'task/{file_name}.h', 'w') as f:
        f.write(input)


d = Data()

d.add(99)

for _ in range(qty - 1):
    # s += f'{random.randint(1,100)},'
    i = random.randint(0,2)
    d.add(i)


write_h("pb99", d.s)
# s += f'{random.randint(1,100)}'


print('done!')

