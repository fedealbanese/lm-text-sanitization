import re
import numpy as np
from torch import nn, topk
from transformers import T5ForConditionalGeneration
from asapp.pfilteringlm import PFilteringModel
from asapp.commons import Unigram


class T5Model(PFilteringModel):
    """
    T5 Language Model, calculates the probability of occurrences of each word in each phrase given its context.

    Attributes:
    model_name(str): name of the HuggingFace model.
    model(T5ForConditionalGeneration): T5 for Conditional Generation HuggingFace model. 
    tokenizer(T5Tokenizer): T5 HuggingFace Tokenizer. 
    softmax(torch.nn.Softmax): softmax function.
    """
    def __init__(
            self,
            model_name: str = "t5-small" #Other models: "t5-base", "t5-large" and "t5-3b".
    ):
        """
        Constructor for the T5 LM.

        Args:
        model_name(str, optional): name of the masking model in HuggingFace.
        """
        super().__init__(model_name = model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.softmax = nn.Softmax(dim=0)
        self.contractions = {
            "re": "are", 
            "ll": "will", 
            "ve": "have", 
            "m": "am",
            "t": "not"
        }
        self.negation_verbs = ["don", "doesn", "isn", "aren", "wasn", "weren", "haven", "hasn", "shouldn", "didn"]

    def is_contraction(self, phrase: str, token_term: str, token_start_i: int) -> bool:
        """
        Returns if the token is a contration.

        Args:
        phrase(str): phrase to process.
        token_term(str): term to mask.
        token_start_i(int): index in phrase where the token starts.

        Returns:  
        (bool): whether the token is a contraction or not.
        """
        return token_start_i !=0 and phrase[token_start_i-1] == "'" and token_term in self.contractions.keys()

    def replace_negation(self, phrase: str) -> str:
        """
        Returns phrase without the sufix of the negation.
        Example "I didn" -> "I did"

        Args:
        phrase(str): phrase to process.

        Returns:  
        phrase(str): phrase with verb without the negation sufix.
        """
        if any(phrase.endswith(negation_verb) for negation_verb in self.negation_verbs):
            return phrase[:-1] #without the final n
        return phrase

    def mask_phrase(
            self,
            phrase: str, 
            token_term: str, 
            token_start_i: int, 
            token_end_i: int, 
            is_contraction: bool, 
            context: str = ""
        ) -> str:
        """
        Returns the original phrase but with the token term masked.

        Args:
        phrase(str): phrase to process.
        token_term(str): term to mask.
        token_start_i(int): index in phrase where the token starts.
        token_end_i(int): index in phrase where the token ends.
        is_contraction(bool): whether the token is a contraction or not.
        context(str, optional): the sanetized previous utterance.

        Returns:  
        masked_phrase(str): phrase with the token masked.
        """
        end_of_beginning_phrase = token_start_i - 1 if is_contraction else token_start_i #remove ' before contractions
        if len(context) == 0: #There is no previous utterance
            beginning_phrase = phrase[:end_of_beginning_phrase]
        else:
            if context[-1] not in ".,:;?!": #Adds point between utterances.
                beginning_phrase = f"{context}.</s> {phrase[:end_of_beginning_phrase]}" 
            else:
                beginning_phrase = f"{context}</s> {phrase[:end_of_beginning_phrase]}"
        if is_contraction and token_term == "t": #it is a negation with contraction.
            beginning_phrase = self.replace_negation(beginning_phrase)
        beginning_phrase = beginning_phrase + " " if is_contraction else beginning_phrase #due to "'" there wasn't a " "
        ending_phrase = phrase[token_end_i:]
        masked_phrase = f"{beginning_phrase}<extra_id_0>{ending_phrase}"
        masked_phrase = self.preprocess(masked_phrase)
        return masked_phrase

    def mask_label(
            self, 
            phrase: str, 
            token_term: str, 
            token_start_i: int, 
            token_end_i: int,  
            is_contraction: bool, 
            context: str = ""
        ) -> str:
        """
        Returns the target label phrase.

        Args:
        phrase(str): phrase to process.
        token_term(str): term to mask.
        token_start_i(int): index in phrase where the token starts.
        token_end_i(int): index in phrase where the token ends.
        is_contraction(bool): whether the token is a contraction or not.
        context(str, optional): the sanetized previous utterance.

        Returns:  
        (str): phrase with the token masked.
        """
        if is_contraction: 
            token_term = self.contractions[token_term]
        if token_start_i == 0 and len(context) == 0:
            if token_end_i == len(context): #The phrase has one token.
                return f"{token_term}"
            else: #First token in phrase.
                return f"{token_term} <extra_id_0>"
        else:
            if token_end_i == len(phrase): #last token in phrase.
                return f"<extra_id_0> {token_term}"
            else: #the token is in the middle of the phrase.
                return f"<extra_id_0> {token_term} <extra_id_1>"

    def calculate_p(self, masked_phrase: str, label_phrase: str, token_start_i: int) -> float:
        """
        Estimates the probability of a token given its context.

        Args:
        phrase(str): phrase to process.
        label_phrase(str): label token phrase.
        token_start_i(int): index in phrase where the token starts.

        Returns:  
        float: probability p of the token.
        """
        input_ids = self.tokenizer(masked_phrase, return_tensors="pt").input_ids.to(self.device)
        labels = self.tokenizer(label_phrase, return_tensors="pt").input_ids.to(self.device)
        output = self.model(input_ids=input_ids, labels=labels)
        token_index = 0 if token_start_i == 0 else 1 #label is the first or second token ("<extra_id_0> token ...")
        probabilities = self.softmax(output.logits[0,token_index,:])
        p = probabilities[labels[0,token_index]].item()
        return p

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
            word_encoding = self.tokenizer(phrase[word.start:word.end], return_tensors='pt').to(self.device)
            word_tokenization = word_encoding["input_ids"][0][:-1] #token ids without end of sentence tokens
            token_start_i = word.start
            for token in word_tokenization:
                token_term = self.tokenizer.decode([token])
                if token_term == "": #GPT2 tokenizer adds "" token after "'"
                    continue
                token_end_i = token_start_i + len(token_term)
                token_is_contraction = self.is_contraction(phrase, token_term, token_start_i)
                masked_phrase = self.mask_phrase(
                    phrase, 
                    token_term, 
                    token_start_i, 
                    token_end_i, 
                    token_is_contraction, 
                    context
                    )
                label_phrase = self.mask_label(
                    phrase, 
                    token_term, 
                    token_start_i, 
                    token_end_i, 
                    token_is_contraction, 
                    context
                    )
                p = self.calculate_p(masked_phrase, label_phrase, token_start_i)
                cumulative_p *= p
                token_start_i = token_end_i #for the next iteration
            word.p = cumulative_p
        return words

    def get_word_embedding(self, word) -> np.ndarray:
        """
        Estimates the word embedding of a term.

        Args:
        word(str): word to calculate its embedding.

        Returns:  
        embedding(np.ndarray): word embedding.
        """ 
        input_id = self.tokenizer(
            word, 
            return_tensors="pt", 
            return_attention_mask=False, 
            add_special_tokens=False
        ).input_ids.to(self.device)
        output = self.model.encoder.embed_tokens(input_id)
        embedding = output[0,0,:].detach().numpy()
        return embedding

    def token_alternatives(
            self, 
            token_term: str, 
            token_start: int, 
            token_end: int, 
            phrase: str, 
            context:str = "", 
            p_threshold:float = 0.01) -> list:
        """
        Calculates top word alternatives for token in phrase.

        Args:
        token_term(str): word to replace.
        token_start(int): index in phrase where the token starts.
        token_end(int): index in phrase where the token ends.
        phrase(str): phrase to process.
        context(str, optional): the sanetized previous utterance.
        p_threshold(float): privacy threshold to use as cutoff. It should be bigger than 0.

        Returns:
        token_alternatives(list): list of strings with words that can replace the original token.
        """
        masked_phrase = self.mask_phrase(
            phrase = phrase, 
            token_term = token_term, 
            token_start_i = token_start, 
            token_end_i = token_end, 
            is_contraction = False, 
            context = context
        )
        label_phrase = self.mask_label(
            phrase = phrase, 
            token_term = token_term, 
            token_start_i = token_start, 
            token_end_i = token_end, 
            is_contraction = False, 
            context = context
        )
        input_ids = self.tokenizer(masked_phrase, return_tensors="pt").input_ids.to(self.device)
        labels = self.tokenizer(label_phrase, return_tensors="pt").input_ids.to(self.device)
        output = self.model(input_ids=input_ids, labels=labels)
        token_index = 0 if token_start == 0 else 1
        probabilities = self.softmax(output.logits[0,token_index,:])

        k = min(100, int(1/p_threshold)) 
        top_k = topk(output.logits[0,token_index,:].flatten(), k).indices
        token_alternatives = []
        for top in top_k:
          if probabilities[top].item() >= p_threshold:
              top_term = self.tokenizer.decode(top)
              if top_term != "":
                  token_alternatives.append(top_term)
          else:
              break
        return token_alternatives