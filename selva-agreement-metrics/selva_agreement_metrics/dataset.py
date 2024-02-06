def read_dataset(filepath):
    with open(filepath) as inpf:
        for line in inpf:
            print(line)
