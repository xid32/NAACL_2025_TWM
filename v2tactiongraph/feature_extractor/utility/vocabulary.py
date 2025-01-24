import spacy
# we do not used this vocabulary class
class Vocabulary:
    PAD_token = 0   # Used for padding short sentences
    BOS_token = 1   # Beginning-of-sentence token
    EOS_token = 2   # End-of-sentence token
    UNK_token = 3   # Unknown word token

    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.PAD_token: "<PAD>", self.BOS_token: "<BOS>", self.EOS_token: "<EOS>", self.UNK_token: "<UNK>"}
        self.num_words = 4
        self.num_sentences = 0
        self.longest_sentence = 0
        self.tokenizer = spacy.load('en_core_web_sm')

    def add_word(self, word):
        if word not in self.word2index:
            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            # Word exists; increase word count
            self.word2count[word] += 1
            
    def add_sentence(self, sentence):
        sentence_len = 0
        for word in self.tokenizer(sentence):
            sentence_len += 1
            self.add_word(str(word))
        if sentence_len > self.longest_sentence:
            # This is the longest sentence
            self.longest_sentence = sentence_len
        # Count the number of sentences
        self.num_sentences += 1
    
    def generate_vector(self, sentence="Hello", longest_sentence=None):
        # Validation data/test data may have longer sentence, so a parameter longest sentence provided
        if longest_sentence is None:
            longest_sentence = self.longest_sentence
        
        vector = [self.BOS_token]
        sentence_len = 0
        for word in self.tokenizer(sentence):
            vector.append(self.to_index(str(word)))
            sentence_len += 1
        vector.append(self.EOS_token)
        
        # Add <PAD> token if needed     
        if sentence_len < longest_sentence:
            for i in range(sentence_len, longest_sentence):
                vector.append(self.PAD_token)
        
        return vector

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        if word not in self.word2index:
            return self.UNK_token
        
        return self.word2index[word]
    
    def filter_vocab(self, min_word_count=0):
        word2count = self.word2count
        self.num_words = 4
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.PAD_token: "<PAD>", self.BOS_token: "<BOS>", self.EOS_token: "<EOS>", self.UNK_token: "<UNK>"}
        for word, count in word2count.items():
            if count>=min_word_count:
                self.word2index[word] = self.num_words
                self.word2count[word] = count
                self.index2word[self.num_words] = word
                self.num_words += 1
