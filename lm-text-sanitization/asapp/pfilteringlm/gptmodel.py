import re
from torch import nn
from transformers import GPT2LMHeadModel
from asapp.pfilteringlm import PFilteringModel
from asapp.commons import Unigram

class GPT2Model(PFilteringModel):
    """
    GPT2 Language Model, calculates the probability of occurrences of each word in each phrase given its context.

    Attributes:
    model_name(str): name of the HuggingFace model.
    prompt_text(str): text added to the beginning of the sentence. GPT2 can not be an empty string.
    initial_text_1(str): text added to the beginning of the context. 
    initial_text_2(str): text added to the beginning of the phrase. 
    initial_index(int): the number of tokens in initial_text or context.
    model(GPT2LMHeadModel): GPT2 HuggingFace model for probability estimation.
    softmax(torch.nn.Softmax): softmax function.
    """
    def __init__(
            self,
            model_name: str = "gpt2", #Other options: "gpt2", "distilgpt2", "gpt2-medium" and "gpt2-large".
            initial_text_1: str = "User 1:",
            initial_text_2: str = "User 2:",
            prompt_text: str = "The following is a conversation between two users. One is an assistant.\n"
    ):
        """
        Constructor for a GPT2 LM.

        Args:
        model_name(str, optional): name of the masking model in HuggingFace.
        prompt_text(str): text added to the beginning of the sentence. GPT2 can not be an empty string.
        initial_text_1(str): text added to the beginning of the context. 
        initial_text_2(str): text added to the beginning of the phrase. 
        """
        super().__init__(model_name = model_name)
        self.prompt_text = prompt_text
        self.prompt_index = self.number_of_tokens(prompt_text)
        self.initial_text_1 = initial_text_1
        self.initial_1_index = self.number_of_tokens(initial_text_1)
        self.initial_text_2 = initial_text_2
        self.initial_2_index = self.number_of_tokens(initial_text_2)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.softmax = nn.Softmax(dim=2)

    def calculate_p(self, phrase: str, context: str = "") -> tuple:
        """
        Calculates the probability of all words in the phrase.

        Args:
        phrase(str): Phrase to process.
        context(str, optional): the sanetized previous utterance.

        Returns: 
        tuple:
            probabilities(torch.Tensor): The probabilities of each token.
            input_ids(torch.Tensor): The id of the tokenization of each words in the input phrase.
        """
        if len(context) == 0: #There is no previous utterance
            complete_phrase = f"{self.prompt_text}{self.initial_text_1} {phrase}"
            self.initial_index = self.prompt_index + self.initial_1_index
        else:
            if context[-1] not in ".,:;?!": #Adds point between utterances.
                complete_phrase = f"{self.prompt_text}{self.initial_text_1} {context}.\n{self.initial_text_2} {phrase}"
                preprocess_phrase = self.preprocess(f"{self.prompt_text}{self.initial_text_1} {context}.\n{self.initial_text_2}")
            else:
                complete_phrase = f"{self.prompt_text}{self.initial_text_1} {context}\n{self.initial_text_2} {phrase}"   
                preprocess_phrase = self.preprocess(f"{self.prompt_text}{self.initial_text_1} {context}\n{self.initial_text_2}")
            self.initial_index = self.number_of_tokens(preprocess_phrase)
        complete_phrase = self.preprocess(complete_phrase)   
        input_ids = self.tokenizer(complete_phrase, return_tensors="pt").input_ids.to(self.device)
        output = self.model(input_ids=input_ids)
        shift_logits = output.logits[..., :-1, :].contiguous() #shift one space the tokens https://github.com/huggingface/transformers/blob/v4.27.2/src/transformers/models/gpt2/modeling_gpt2.py#L1102
        probabilities = self.softmax(shift_logits)
        return probabilities, input_ids

    def number_of_tokens(self, text: str) -> int:
        """
        Returns the number of tokens in text.

        Args:
        text(str): phrase to tokenize.

        Returns: 
        n_tokens(int): number of tokens in text.
        """
        n_tokens = len(self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)[0]) 
        return n_tokens

    def add_word_p(self, words, probabilities, input_ids, phrase) -> list:
        """
        Adds each word's probability to the list of unigrams.

        Args:
        words(list of Tokens): representation of the words in document without attribute p.
        probabilities(torch.Tensor): The probabilities of each token.
        input_ids(torch.Tensor): The id of the tokenization of each words in the input phrase.
        phrase(str): the phrase to process.

        Returns: 
        words(list of Tokens): representation of the words in document with attribute p.
        """
        #skip initial_text tokens.
        token_index = self.initial_index
        for word in words:
            cumulative_p = 1
            n_characters_matched = 0 #how manny characters of the word were matched by subtokens
            while len(word.term) > n_characters_matched:
                token_term = self.tokenizer.decode(input_ids[0][token_index]).replace(' ', '')
                word_to_match = word.term[n_characters_matched:]
                if word_to_match.startswith(token_term.replace("'","").lower()) and token_term != "'":	
                    n_characters_matched += len(token_term.replace("'",""))
                    cumulative_p *= probabilities[0, token_index-1, input_ids[0][token_index]].item()
                token_index += 1
            word.p = cumulative_p
        return words

    def get_probabilities(self, phrase: str, words: list, context: str = "") -> list:
        """
        Estimates the probability of the words and adds each words probability to the words representation.

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
        probabilities, input_ids = self.calculate_p(phrase, context)
        words = self.add_word_p(words, probabilities, input_ids, phrase)
        return words
