import random
import os

order = [i for i in range(13)]
random.shuffle(order)
# make txt order
with open("./order.txt", 'w', encoding='utf-8') as f:
    dir_list = os.listdir(".")
    for idx in order:
        filename = dir_list[idx]
        if filename.endswith(".py"):
            continue
        f.write(f"{filename}\n")