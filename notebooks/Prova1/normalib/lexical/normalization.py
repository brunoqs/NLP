import nltk
import unidecode
import string

class Normalization:

    def __init__(self):
        self.sent_tokenize = nltk.data.load('tokenizers/punkt/portuguese.pickle')
        self.stopwords = nltk.corpus.stopwords.words('portuguese') # dicionario de stopwords portugues
        self.stemmer = nltk.stem.RSLPStemmer() # stemmer para lingua portuguesa
        self.lemmatizer = nltk.stem.WordNetLemmatizer() # lemmatizer apenas para lingua inglesa

    def to_lower_case(self, text):
        return text.lower()

    def remove_accents(self, text):
        return unidecode.unidecode(text)

    def remove_punctuation(self, text):
        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        without_punctuation = tokenizer.tokenize(text)
        return ' '.join(without_punctuation)

    # remove stopwords e retorna um string para facilitar no pipeline
    def remove_stopwords(self, text):
        tokens = self.tokenize_words(text)
        without_stopwords = [token for token in tokens if token not in self.stopwords]
        return ' '.join(without_stopwords)

    def tokenize_sentences(self, text):
        sentences = self.sent_tokenize.tokenize(text)
        return sentences

    def tokenize_words(self, text):
        tokens = nltk.tokenize.word_tokenize(text)
        return tokens

    # nltk aceita apenas lingua inglesa
    def lemmatize(self, text):
        tokens = self.tokenize_words(text)
        lemmas = [self.lemmatizer.lemmatize(lemma) for lemma in tokens]
        return lemmas

    # retorna todas as palavras raizes como uma string para facilitar no pipeline
    def stemmize(self, text):
        tokens = self.tokenize_words(text)
        stems = [self.stemmer.stem(stem) for stem in tokens]
        return ' '.join(stems)

    def normalization_pipeline(self, text, to_lower_case=False, remove_accents=False, remove_punctuation=False, remove_stopwords=False,
                    lemmatize=False, stemmize=False, tokenize_sentences=False, tokenize_words=False):
        text = self.to_lower_case(text) if to_lower_case else text               
        text = self.remove_accents(text) if remove_accents else text
        text = self.remove_punctuation(text) if remove_punctuation else text
        text = self.remove_stopwords(text) if remove_stopwords else text
        text = self.lemmatize(text) if lemmatize else text
        text = self.stemmize(text) if stemmize else text
        text = self.tokenize_sentences(text) if tokenize_sentences else text
        text = self.tokenize_words(text) if tokenize_words else text
        
        return text

