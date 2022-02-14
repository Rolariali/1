import base64
import json
import os
import random
import sys


qty = int(sys.argv[1])

s = ""

for i in range(qty - 1):
    s += f'{random.randint(1,100)},'

s += f'{random.randint(1,100)}'


input = "std::vector<uint8_t> input = { " +  s  + " };"

# print(input)
with open('task/test_data.h', 'w') as f:
    f.write(input)

print('done')

