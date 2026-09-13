import random

# Manually set CHB, CEU and YRI- could make this a param later
pop = "CHB"

random.seed(42)

with open(f'{pop}.txt') as file:
    data = [line.strip() for line in file]

num_subsamples = 10
subsample_count = [5, 10, 15, 20]

for x in subsample_count:
    for i in range(num_subsamples):
        random_subset = random.sample(data, x)
        with open(f'{pop}_subsamples/{x}_individuals/{pop}_subsample{x}_replicate{i+1}.txt', 'w') as file:
            for sample in random_subset:
                file.write(str(sample) + '\n')
