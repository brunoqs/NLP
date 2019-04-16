import os

from normalib.lexical import Normalization


corpus_path = "Crawler/data/corpora/tecnologia/"
files = [f for f in os.listdir(corpus_path) if os.path.isfile(os.path.join(corpus_path, f))]

corpus_lines = []
for file in files:
    with open(os.path.join(corpus_path, file), "r") as text_file:
        lines = text_file.readlines()
        corpus_lines.extend(lines)

n = Normalization()

print(n.normalization_pipeline(' '.join(corpus_lines), remove_accents=True, remove_punctuation=True, 
                                tokenize_sentences=False, tokenize_words=False, lemmatize=True, stemmize=True))