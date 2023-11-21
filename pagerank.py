import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    if page not in corpus or len(corpus[page]) == 0:
        return {p: 1 / len(corpus) for p in corpus}

    num_links = len(corpus[page])
    damping_prob = damping_factor / num_links
    nondamping_prob = (1 - damping_factor) / len(corpus)

    # Initialisation de la probabilité de distribution
    transition_probabilities = {p: nondamping_prob for p in corpus}
    
    # Mise à jour des probabilités pour les pages
    for linked_page in corpus[page]:
        transition_probabilities[linked_page] += damping_prob

    return transition_probabilities


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    raise NotImplementedError


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    N = len(corpus)
    pageranks = {page: 1 / N for page in corpus}

    # Itération jusqu'à convergence
    while True:
        new_pageranks = {}  # Nouvelles valeurs de PageRank
        # Calcul des nouvelles valeurs de PageRank pour chaque page
        for page in corpus:
            #Initialisation
            new_pagerank = ( 1 - damping_factor ) / N 

            for nom_page in corpus:
                # on check si la page fait parti d'un ensemble pour incrémenter la pagerank avec la formule
                if page in corpus[nom_page]:
                    n_link = len(corpus[nom_page])
                    new_pagerank = new_pagerank + damping_factor * pageranks[nom_page] / n_link
            #Maj du nouveau Pagerank
            new_pageranks[page] = new_pagerank

        # test de la convergence
        convergence = all(abs(new_pageranks[page] - pageranks[page]) < 0.001 for page in corpus)
        # prendre les nouveaux page rank
        pageranks = new_pageranks
        # On sort si la convergence est atteinte
        if convergence:
            break
    return pageranks


if __name__ == "__main__":
    main()
