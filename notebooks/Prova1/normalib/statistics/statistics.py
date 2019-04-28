from normalib.lexical import Normalization
import nltk
import heapq

class Statistics:

    def __init__(self):
        self.n = Normalization()

    def frequently_words(self, text, n=20, more=True):
        frequency = nltk.probability.FreqDist(word.lower() for word in self.n.tokenize_words(text))
        if more:
            most_frequently = heapq.nlargest(n, frequency, frequency.get)
            return most_frequently
        else:
            less_frequently = heapq.nsmallest(n, frequency, frequency.get)
            return less_frequently

    def average_size_words(self, text):   
        size_words = [len(word.lower()) for word in self.n.tokenize_words(text)]
        average = sum(size_words) / len(size_words)
        return average

    def average_size_sentences(self, text):   
        size_sentences = [len(sentence.lower().split()) for sentence in self.n.tokenize_sentences(text)]
        average = sum(size_sentences) / len(size_sentences)
        return average

    def word_appear(self, text, word):
        frequency = nltk.probability.FreqDist(word.lower() for word in self.n.tokenize_words(text))
        return frequency[word]

    def word_once_length(self, text):
        frequency = nltk.probability.FreqDist(word.lower() for word in self.n.tokenize_words(text))
        return len(frequency.hapaxes())
