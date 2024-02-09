import re
from numpy.random import randint, seed
from asapp.commons import Redactor
from asapp.commons import Unigram


class RedactorReplace(Redactor):
    """
    Redacts terms in a document that don't meet a privacy threshold and can not be replaced.

    Attributes:
    context(str): the sanetized previous utterance.
    """
    def __init__(self, model, p = 0.0, replacement_p = 1e-50, s = 0.0, k = 50):
        """
        Constructor for a Redactor.

        Args:
        model: language model to query for word frequencies.
        p(float): privacy threshold to use as cutoff.
        replacement_p(float): privacy threshold to use as cutoff for replacement.
        s(float): similarity thershold to use as cutoff of alternative tokes.
        k(int): number of alternative tokens to consider.
        """
        super().__init__( model, p)
        self.replacement_p_threshold = replacement_p
        self.s_threshold = s
        self.k_alternatives = k
        self.context = ""

    def is_safe(self, token: Unigram, document: str) -> bool:
        """
        Returns whether a token is safe based on its probability and its similarity with alternative words.

        Args:
        token(Unigram): token to process.
        document(str): document to redact.

        Returns: 
        bool: rwhether a token is safe.
        """
        if token.p >= self.p_threshold:
            return True
        else:
            token_alternatives = self.model.token_alternatives(
                token.term, 
                token.start, 
                token.end, 
                document, 
                self.context,
                self.p_threshold) #list of token alternatives with p bigger than p_threshold.
            for token_alternative in token_alternatives: #TODO: calculate all distances at the same time. It is faster.
                if (len(token_alternative) > 2  and token_alternative[0] != "<" and token_alternative[0] != "["): 
                    similarity = self.model.token_similarity(token.term, token_alternative)
                    is_alternative = similarity <= self.s_threshold #whether the distance between token and alternative is smaller than s_threshold.
                    if is_alternative: #if it found a valid alternative
                        return True
        return False
    
    def update_replaced_text(self, text:str, start:int, end:int, replaced_text:str, index_t:int, token_alternative:str):
        """
        Adds to the replaced_text the safe text before a redaction and a token alternative.

        Args:
        text(str): text of the utterance.
        start(int): start index of the unsafe word.
        end(int): end index of the unsafe word.
        replaced_text(str): safe text of the utterance.
        index_t(int): index of the last unsafe word in text.
        token_alternative(str): replacement token.
        
        Returns:
        replaced_text(str): updated safe text of the utterance.
        index_t(int): updated index of the last unsafe word in text.
        """
        replaced_text += text[index_t:start]
        replaced_text += token_alternative
        index_t = end
        return replaced_text, index_t

    def is_numeric(self, input_text:str) -> bool:
        """
        Returns whether the term is a number or not.

        Args:
        input_text(str): text to process.

        Returns:
        bool: whether the term is a number or not.
        """
        return all(char.isdigit() for char in input_text)

    def replace_numbers(self, input_text:str) -> str:
        """
        Returns a string with the digits randomly replaced by other digits.

        Args:
        input_text(str): text to process.

        Returns:
        output_text(str): input_text but with the digits randomly replaced.
        """
        seed(0) #set seed.
        output_text = ""
        for char in input_text:
            if char.isdigit():
                output_text = output_text + str(randint(10))  
            else:
                output_text = output_text + char
        return output_text
    
    def find_replacement( #TODO hoy: acortar nombre y pasar a la clase redactor replace. Tambien agregar algo que haga todo el for.
        self,
        redaction:list,
        text:str,
        context:str,
        s:float,
        index_t:int,
        replaced_text:str,
        model,
        all_replacements:dict,
        utterance_replacements:dict,
        default_replacement:str = "***"): 
        """
        Process an utterance and finds a replacement word for a redations.

        Args:
        redaction(list): list of redactions.
        text(str): text of the utterance.
        context(str): previous utterance.
        s(float): similarity threshold.
        index_t(int): index in text for constructing replaced_text without unsafe words.
        replaced_text(str): text with unsafe words replaced by alternative tokens.
        model: pfilteringlm model to calculate similar words.
        all_replacements(dict): dictionary with all pairs unsafe_word: alternative_word
        utterance_replacements(dict): dictionary with pairs unsafe_word: alternative_word that was use in this utterance.
        default_replacement(str): default replacement in case there are no alternative tokens with similarity bigger than s.
        
        Returns:
        replaced_text(str): updated text with unsafe words replaced by alternative tokens.
        index_t(int): updated index in text for constructing replaced_text without unsafe words.
        all_replacements(dict): updated dictionary with all pairs unsafe_word: alternative_word
        utterance_replacements(dict): updated dictionary with pairs unsafe_word: alternative_word that was use in this utterance.
        """
        tag, start, length = redaction
        tag = tag[10:] # remove the "PFILTERED:" in "PFILTERED:word"
        end = start + length
        redaction_text = text[start:end]
        redaction_term = next(Unigram.tokenize(redaction_text)).term
        if redaction_term in all_replacements:
            utterance_replacements[redaction_term] = all_replacements[redaction_term]
            replaced_text, index_t = self.update_replaced_text(
                text, 
                start, 
                end, 
                replaced_text, 
                index_t, 
                token_alternative = utterance_replacements[redaction_term]
            )
        elif self.is_numeric(redaction_term):
            token_alternative = self.replace_numbers(redaction_text)
            all_replacements[redaction_term] = token_alternative
            utterance_replacements[redaction_term] = token_alternative
            replaced_text, index_t = self.update_replaced_text(
                text, 
                start, 
                end, 
                replaced_text, 
                index_t, 
                token_alternative
            )
        else:
            #find alternative
            best_token_alternative = default_replacement
            token_alternatives = model.token_alternatives(
                redaction_text,
                start, 
                end, 
                text, 
                context, 
                k = self.k_alternatives,
                p_threshold = self.replacement_p_threshold
            ) 
            for token_alternative in token_alternatives: #TODO: calculate all distances at the same time. It is faster.
                unuseful_alternatives = [redaction_term, default_replacement, "", "(", ")", "[", "]"]
                if (
                    token_alternative not in unuseful_alternatives and
                    len(token_alternative) > 2  and 
                    token_alternative[0] != "<" and 
                    token_alternative[0] != "["
                ):
                    similarity = model.token_similarity(redaction_text, token_alternative)
                    if similarity <= s: #if it found a valid alternative
                        s = similarity
                        best_token_alternative = token_alternative
            all_replacements[redaction_term] = best_token_alternative
            utterance_replacements[redaction_term] = best_token_alternative
            replaced_text, index_t = self.update_replaced_text(
                text, 
                start, 
                end, 
                replaced_text, 
                index_t, 
                best_token_alternative
            )
        return replaced_text, index_t, utterance_replacements, all_replacements
    
    def find_all_replacements(
        self,
        redactions:list,
        text:str,
        context:str,
        s:float,
        model,
        all_replacements:dict,
        default_replacement:str = "***"
        ):
        """
        Process an utterance and finds a replacement word for a redations.

        Args:
        redactions(list): list of redactions.
        text(str): text of the utterance.
        context(str): previous utterance.
        s(float): similarity threshold.
        model: pfilteringlm model to calculate similar words.
        all_replacements(dict): dictionary with all pairs unsafe_word: alternative_word
        default_replacement(str): default replacement in case there are no alternative tokens with similarity bigger than s.
        
        Returns:
        replaced_text(str): updated text with unsafe words replaced by alternative tokens.
        index_t(int): updated index in text for constructing replaced_text without unsafe words.
        all_replacements(dict): updated dictionary with all pairs unsafe_word: alternative_word
        utterance_replacements(dict): updated dictionary with pairs unsafe_word: alternative_word that was use in this utterance.
        """
        utterance_replacements = {} #key:redacted_token. value: replace_token
        replaced_text = "" #utterance text with all the replacements
        index_t = 0
        for redaction in redactions:
            replaced_text, index_t, utterance_replacements, all_replacements = self.find_replacement(
                redaction,
                text,
                context,
                s,
                index_t,
                replaced_text,
                model,
                all_replacements,
                utterance_replacements,
                default_replacement
            )
        return replaced_text, index_t, utterance_replacements, all_replacements 
