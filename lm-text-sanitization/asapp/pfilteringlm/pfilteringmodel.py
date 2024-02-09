import torch
import pickle
from transformers import AutoTokenizer
from asapp.commons import Unigram
from scipy.spatial.distance import cosine


class PFilteringModel:
    """
    PFilteringModel Model.

    Attributes:
    model_name(str): name of the HuggingFace model.
    device(str): device to run HuggingFace model.
    """
    def __init__(
            self,
            model_name: str
    ):
        """
        Constructor for a LM.

        Args:
        model_name(str, optional): name of the masking model in HuggingFace.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.word2embedding_data = {}

    def upload_word2embedding_file(self, file_name:str):
        """
        Loads file and update word2embedding_data,

        Args:
        file_name(str): path to word2embedding_data file with a dicctionary.
        """
        with open(file_name, 'rb') as handle:
            self.word2embedding_data = pickle.load(handle)

    def dump_word2embedding_file(self, file_name:str):
        """
        Dump update word2embedding_data to a file,

        Args:
        file_name(str): path to save word2embedding_data.
        """
        with open(file_name, 'wb') as handle:
            pickle.dump(self.word2embedding_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_words(self, phrase: str) -> list:
        """
        Returns the list of words in the phrase.

        Args:
        phrase(str): phrase to process.

        Returns:  
        token_doc_representation(list of Tokens): representation of the document. 
        Each token has:
            token.term(str): the characters it represents.
            token.start(int): the index of the document where the token begins.
            token.end(int): the index of the document where the token ends.
        """
        token_doc_representation = []
        matches = Unigram._token_regex.finditer(phrase) 	
        for match in matches: 
            unigram = Unigram(match)
            token_doc_representation.append(unigram)
        return token_doc_representation

    def preprocess(self, phrase: str) -> str:
        """
        Preprocess a phrase. It adds a final point and fixes double and triple spaces.

        Args:
        phrase(str): phrase to preprocess.

        Returns:
        phrase(str): phrase with final point and without double and triple spaces.
        """
        if len(phrase) != 0  and phrase[-1] not in ".,:;?!_": #Add final point.
            phrase = phrase + "."
        phrase = phrase.replace("  ", " ") #fix double spaces.
        phrase = phrase.replace("   ", " ") #fix triple spaces.
        return phrase  

    def get_doc_repr(self, phrase: str, context: str = "") -> list:
        """
        Returns the representation of a document.

        Args:
        phrase(str): phrase to process.
        context(str, optional): the sanetized previous utterance.

        Returns:
        token_doc_representation(list of Tokens): representation of the document. 
        Each token has:
            token.term(str): the characters it represents.
            token.start(int): the index of the document where the token begins.
            token.end(int): the index of the document where the token ends.
            token.p(float): the probability of the token occurence given its context according to the language model. 

        Raises:
        if input phrase has masking string.    
        """
        words = self.get_words(phrase)
        token_doc_representation = self.get_probabilities(phrase, words, context)
        return token_doc_representation

    def token_similarity(self, word_1: str, word_2: str) -> float:
        """
        Calculates the cosine similarity between the embeddings of word_1 and word_2.

        Args:
        word_1(str): word to calculate the cosine similarity.
        word_2(str): word to calculate the cosine similarity.

        Returns:  
        similarity(float): cosine similarity between the embeddings of word_1 and word_2.
        """
        embedding_1 = self.get_word_embedding(word_1)
        embedding_2 = self.get_word_embedding(word_2)
        similarity = cosine(embedding_1, embedding_2)
        return similarity
    