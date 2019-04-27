import os
from normalib.lexical import Normalization
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument

# Testar biblioteca normalib
corpus_path = "Crawler/data/corpora/tecnologia/"
files = [f for f in os.listdir(corpus_path) if os.path.isfile(os.path.join(corpus_path, f))]

corpus_lines = []
for file in files:
    with open(os.path.join(corpus_path, file), "r") as text_file:
        lines = text_file.readlines()
        corpus_lines.extend(lines)

n = Normalization()

all_sentences = n.normalization_pipeline(' '.join(corpus_lines), to_lower_case=True, remove_accents=False, remove_punctuation=True, 
                                remove_stopwords=True, lemmatize=False, stemmize=False, tokenize_sentences=False, tokenize_words=False)
print(n.frequently_words(all_sentences))    
print(n.word_appear(all_sentences, 'é'))                      


# Word2Vec e Doc2Vec
# n = Normalization()

# corpora_path = 'Crawler/data/corpora/'
# files_tecnologia = os.listdir('{}/tecnologia/'.format(corpora_path))
# files_tecnologia = ['{}/tecnologia/{}'.format(corpora_path,f) for f in files_tecnologia if f != '.DS_Store']
# files_mercado = os.listdir('{}/mercado/'.format(corpora_path))
# files_mercado = ['{}/mercado/{}'.format(corpora_path,f) for f in files_mercado if f != '.DS_Store']

# all_sentences = []
# for file in files_tecnologia:
#     with open(file, 'r') as text_file:
#         lines = text_file.readlines()
#         for line in lines:
#             line = n.to_lower_case(line)
#             sentences = n.tokenize_sentences(line)
#             sentences = [n.tokenize_words(sent) for sent in sentences]
#             all_sentences.extend(sentences)

# print("Number of sentences: {}".format(len(all_sentences)))


# w2vmodel_tecnologia = Word2Vec(all_sentences, size=200, window=5, min_count=3, workers=4)
# print(w2vmodel_tecnologia.wv.most_similar('linux'))



# Doc2Vec
# all_documents = []
# all_files = files_tecnologia
# all_files.extend(files_mercado)
# for file in all_files:
#     with open(file, 'r') as text_file:
#         document = ' '.join(text_file.readlines())
#         document = n.to_lower_case(document)
#         document_tokens = n.tokenize_words(document)
#         all_documents.append(document_tokens)
# print("Number of documents: {}".format(len(all_documents)))
# tagged_documents = [TaggedDocument(words=d, tags=[str(i)]) for i, d in enumerate(all_documents)]

# d2vmodel = Doc2Vec(tagged_documents, vector_size=20, window=2, min_count=1, workers=4)

# vector_tec = d2vmodel.infer_vector(all_documents[0])
# vector_tec2 = d2vmodel.infer_vector(all_documents[1])
# vector_pol = d2vmodel.infer_vector(all_documents[len(all_documents)-1])
# vector_pol2 = d2vmodel.infer_vector(all_documents[len(all_documents)-2])

# from scipy import spatial

# print(1 - spatial.distance.cosine(vector_pol, vector_tec))
# print(1 - spatial.distance.cosine(vector_pol, vector_pol2))
# print(1 - spatial.distance.cosine(vector_tec, vector_tec2))