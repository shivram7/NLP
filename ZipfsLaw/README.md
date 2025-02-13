# Zipf's Law Verification  

This project aims to **demonstrate Zipf's Law** by analyzing the frequency distribution of words in a given text file and plotting a **rank-frequency distribution** on a logarithmic scale.  

## What is Zipf's Law?  
Zipf's Law states that **the frequency of a word is inversely proportional to its rank in the frequency table**. That is, the most frequent word appears approximately **twice** as often as the second most frequent word, **three times** as often as the third most frequent word, and so on. When plotted on a **log-log scale**, this results in a **straight line**.

## Dependencies  

Ensure you have Python installed along with the required libraries:  

```bash
pip install matplotlib
```

##  How to Run  

Run the script using:  

```bash
python zipfs_law.py --path path/to/textfile.txt
```

where `path/to/textfile.txt` is the input text file to analyze.  

##  How It Works  

1. The script reads the input text file and counts **word frequencies** using `Counter` from the `collections` module.  
2. It ranks the words based on frequency and stores the **rank-frequency pairs**.  
3. The script then **plots a log-log graph** of rank vs. frequency to visually confirm Zipf’s Law.  

##  Example Output  

The script generates a **log-log plot** of word frequency vs. rank and saves it as `graph.png`. The expected output is a **straight-line trend**, confirming Zipf’s Law.  

##  Functions Overview  

| Function | Description |  
|----------|------------|  
| `get_ranks_and_frequencies(infile)` | Reads a text file and returns word rank-frequency pairs. |  
| `plot(infile)` | Generates a **log-log plot** of word rank vs. frequency and saves it. |  

##  Conclusion  

This script effectively visualizes Zipf’s Law by analyzing text data and demonstrating the expected **power-law distribution** of word frequencies.  
