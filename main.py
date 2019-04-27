import os

from normalib.lexical import Normalization

from gensim.models import Word2Vec, Doc2Vec


corpus_path = "Crawler/data/corpora/tecnologia/"
files = [f for f in os.listdir(corpus_path) if os.path.isfile(os.path.join(corpus_path, f))]

corpus_lines = []
for file in files:
    with open(os.path.join(corpus_path, file), "r") as text_file:
        lines = text_file.readlines()
        corpus_lines.extend(lines)

n = Normalization()

all_sentences = n.normalization_pipeline(' '.join(corpus_lines), to_lower_case=True, remove_accents=True, remove_punctuation=True, 
                                remove_stopwords=True, tokenize_sentences=False, tokenize_words=False, lemmatize=False, stemmize=True)
print(all_sentences)                        
# w2vmodel_tecnologia = Word2Vec(all_sentences, size=200, window=5, min_count=3, workers=4)
# w2vmodel_tecnologia.wv.most_similar('fedora')
