import re
import transformers
import numpy as np
import tensorflow as tf
from transformers import pipeline, TFBertModel
from asapp.pfilteringlm import PFilteringModel
from asapp.commons import Unigram


class FillMaskModel(PFilteringModel):
    """
    Fill Mask Language Model, calculates the probability of occurrences of each word in each phrase given its context.

    Attributes:
    model_name(str): name of the HuggingFace model.
    mask_str(str): masking string.
    model(FillMaskPipeline): filling mask HuggingFace model for probability estimation.
    prompt_text(str): text added to the beginning of the sentence. GPT2 can not be an empty string.
    initial_text_1(str): text added to the beginning of the context. 
    initial_text_2(str): text added to the beginning of the phrase. 
    """
    def __init__(
            self,
            model_name: str = "bert-base-uncased", #Other models: "roberta-base" and "albert-base-v2".
            initial_text_1: str = "Assistant: ",
            initial_text_2: str = "User: ",
            prompt_text: str = "This is a dialogue.\n"
    ):
        """
        Constructor for a fill mask LM.

        Args:
        model_name(str, optional): name of the masking model in HuggingFace.
        prompt_text(str): text added to the beginning of the sentence. GPT2 can not be an empty string.
        initial_text_1(str): text added to the beginning of the context. 
        initial_text_2(str): text added to the beginning of the phrase. 
        """
        super().__init__(model_name = model_name)
        if float(transformers.__version__[0:4]) < 4.21:
            self.device = 0 if self.device == "cpu" else 1
        self.model = pipeline('fill-mask', model = model_name, device=self.device)
        self.mask_str =  self.tokenizer.mask_token
        self.initial_text_1 = initial_text_1
        self.initial_text_2 = initial_text_2
        self.prompt_text = prompt_text
        self.model_for_embeddings = TFBertModel.from_pretrained(model_name,  from_pt = True)

    def mask_phrase(
            self, 
            phrase: str, 
            token_term: str, 
            token_start_i: int, 
            token_end_i: int, 
            context: str = ""
            ) -> str:
        """
        Returns the original phrase but with the token term masked.

        Args:
        phrase(str): phrase to process.
        token_term(str): term to mask.
        token_start_i(int): index in phrase where the token starts.
        token_end_i(int): index in phrase where the token ends.
        context(str, optional): the sanetized previous utterance.

        Returns:  
        masked_phrase(str): phrase with the token masked.

        Raises:
        if input phrase has masking string.
        """
        if self.mask_str in phrase:
            print("Error: input phrase has a {}.".format(self.mask_str))
            raise
        if len(context) == 0: #There is no previous utterance
            beginning_phrase = f"{self.prompt_text} [SEP] {self.initial_text_1} [SEP] {phrase[:token_start_i]}"
        else:
            if context[-1] not in ".,:;?!": #Adds point between utterances.
                beginning_phrase = f"{self.prompt_text} [SEP] {self.initial_text_1} [SEP] {context}. [SEP] {self.initial_text_2} [SEP] {phrase[:token_start_i]}"
            else:
                beginning_phrase = f"{self.prompt_text} [SEP] {self.initial_text_1} [SEP] {context} [SEP] {self.initial_text_2} [SEP] {phrase[:token_start_i]}"
        ending_phrase = phrase[token_end_i:]
        masked_phrase = f"{beginning_phrase}{self.mask_str}{ending_phrase}"
        masked_phrase = self.preprocess(masked_phrase)
        if len(self.tokenizer(masked_phrase)['input_ids']) > 511: #context too long.
            beginning_phrase = f"{self.initial_text_1} [SEP] {context} [SEP] {self.initial_text_2} [SEP] {phrase[:token_start_i]}"
            masked_phrase = f"{beginning_phrase}{self.mask_str}{ending_phrase}"
            masked_phrase = self.preprocess(masked_phrase)
        return masked_phrase

    def calculate_p(self, masked_phrase: str, token_term: str) -> float:
        """
        Estimates the probability of a token given its context.

        Args:
        phrase(str): phrase to process.
        token_term(str): term to mask.

        Returns:  
        float: probability p of the token.
        """
        if token_term == "": #Albert tokenizer adds "" token after "."
            return 1
        p = self.model(masked_phrase, targets = [token_term])
        return p[0]["score"]

    def get_probabilities(self, phrase: str, words: list, context: str = "") -> list:
        """
        Estimates the probability of the words given its context.

        Args:
        phrase(str): phrase to process.
        words(list of Unigram): list of words to mask and calculate p.
        context(str, optional): the sanetized previous utterance.

        Returns:  
        words(list of Unigram): same as input but with probabilities p. 
        Each token has:
            token.term(str): the characters it represents.
            token.start(int): the index of the document where the token begins.
            token.end(int): the index of the document where the token ends.
            token.p(float): the probability of the token occurence given its context according to the language model. 

        """
        for word in words: 
            cumulative_p = 1
            word_encoding = self.tokenizer(word.term, return_tensors='pt')
            word_tokenization = word_encoding["input_ids"].to(self.device)[0][1:-1] #token ids with start and end of sentence tokens
            token_start_i = word.start
            for token in word_tokenization:
                token_term = self.tokenizer.decode([token])
                token_end_i = token_start_i + len(token_term.replace("##", "")) #subwords tokens start with ##
                masked_phrase = self.mask_phrase(phrase, token_term, token_start_i, token_end_i, context)
                p = self.calculate_p(masked_phrase, token_term)
                cumulative_p *= p
                token_start_i = token_end_i #for the next iteration
            word.p = cumulative_p
        return words

    def get_word_embedding(self, word:str) -> np.ndarray:
        """
        Estimates the word embedding of a term.
        Based in https://stackoverflow.com/questions/74996994/do-bert-word-embeddings-change-depending-on-context

        Args:
        word(str): word to calculate its embedding.

        Returns:  
        word_embedding(np.ndarray): word embedding.
        """ 
        if word in self.word2embedding_data:
            return self.word2embedding_data[word]
        else:
            word_token = tf.constant(self.tokenizer.encode(word))[None,:]
            embeddings = self.model_for_embeddings(word_token) #TODO: calculate word embedding based on complete phrase and not only the word.
            word_embedding = embeddings[0][0,1,:] #index=[0,1,:] because: [CLS] word [SEP]
            word_embedding = word_embedding.numpy().reshape(1,-1)[0] #reshape
            self.word2embedding_data[word] = word_embedding
            return word_embedding

    def token_alternatives(
            self, 
            token_term: str, 
            token_start: int, 
            token_end: int, 
            phrase: str, 
            context:str = "", 
            p_threshold:float = 0.01,
            k:int = 10) -> list:
        """
        Calculates top word alternatives for token in phrase.

        Args:
        token_term(str): word to replace.
        token_start(int): index in phrase where the token starts.
        token_end(int): index in phrase where the token ends.
        phrase(str): phrase to process.
        context(str, optional): the sanetized previous utterance.
        p_threshold(float): privacy threshold to use as cutoff.
        k(int): number of alternative tokens to consider.

        Returns:
        token_alternatives(list): list of strings with words that can replace the original token.
        """
        masked_phrase = self.mask_phrase(phrase, token_term, token_start, token_end, context)
        top_k_words = self.model(masked_phrase, top_k=k)

        token_alternatives = []
        for top in top_k_words:
          if top["score"] >= p_threshold:
              top_term = top["token_str"]
              if top_term != "":
                  token_alternatives.append(top_term)
          else:
              break
        return token_alternatives
