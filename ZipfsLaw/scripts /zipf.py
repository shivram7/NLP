import argparse
from collections import Counter
from matplotlib import pyplot as plt
import math

def get_ranks_and_frequencies(infile):

    with open(infile) as f:
        contents = f.read()
    c = Counter(contents.split())
    #c_sort = sorted(c.items(), key=lambda x: x[1], reverse=True)
    #print(c)

    ranks_and_frequencies = [(rank,val) for rank,val in enumerate(sorted(c.values(), reverse=True), start=1)]
    #print(ranks_and_frequencies)
    return ranks_and_frequencies

def plot(infile):

    ranks_and_frequencies = get_ranks_and_frequencies(infile) 
    #print(ranks_and_frequencies)
    ranks = [rank for rank,x in ranks_and_frequencies]
    frequencies = [freq for x,freq in ranks_and_frequencies]

    plt.figure(figsize=(10, 6))
    plt.loglog(ranks,frequencies)
    plt.xlabel('Rank (log scale)', fontsize=12)
    plt.xlim(0, max(ranks)+50)
    plt.ylabel('Frequency (log scale)', fontsize=12)
    plt.title('Shivani Ramesh', fontsize=14)
    plt.savefig("graph.png")

    plt.show()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Constructs a curve '
    'demonstrating Zipf\'s Law '
    'by plotting a rank, '
    'frequency plot.')
    parser.add_argument('--path', type=str, required=True, help='Path to file')
    args = parser.parse_args()
    plot(args.path)


