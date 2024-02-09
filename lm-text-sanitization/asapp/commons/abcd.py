
import json
import re


class ABCD:  # TODO: Create a dataset interface common for ABCD and TAB.
    """
    This class wraps the ABCD dataset, and exports common operations.
    """

    # A set of blatantly safe terms included in "unsafe" metadata
    blatantly_safe = frozenset({
        "as", "in", "and", "or", "but", "the", "then", "a", "at", "dot", "com",
        "street", "st", "avenue", "ave", "av", "drive", "drv"
    })

    whitespace_regex = re.compile(r"\s+")

    def __init__(self, abcd_filename):
        """
        Constructor for an ABCD dataset.
        :param abcd_filename: str, filename from where to load the dataset.
        """
        self.abcd_filename = abcd_filename
        self.abcd_data = {}
        self.load()

    def load(self):
        """
        Loads the dataset from file.
        """
        with open(self.abcd_filename) as abcd_file:
            self.abcd_data = json.load(abcd_file)

    def dump(self, output_filename=None):
        """
        Dumps the dataset to a file
        :param output_filename: alternative file where to dump contents.
        """
        if output_filename is None:
            output_filename = self.abcd_filename
        with open(output_filename, 'w') as abcd_file:
            json.dump(self.abcd_data, abcd_file)

    def iter_entry_utterances(self, entry, scrub_regex=None):
        """
        Returns an iterator over the utterances in an ABCD entry.
        :param entry: dict, ABCD entry.
        :param scrub_regex: optional regex used to remove substrings.
        :return: str, utterance in the entry.
        """
        for utterance_data in entry['original']:
            sender, utterance = utterance_data
            if sender == 'action':
                continue
            if scrub_regex:
                utterance = scrub_regex.sub('', utterance).strip()
            if not utterance:
                continue
            yield utterance

    def iter_utterances(self, category):
        """
        Returns an iterator over the utterances in an ABCD category.
        :param category: str, ABCD category (e.g., 'train', 'dev', 'test').
        :return: iterator(string), iterator to utterances in the category.
        """
        recurrent_data_regex = None
        for entry in self.abcd_data[category]:
            yield from self.iter_entry_utterances(entry, recurrent_data_regex)

    def iter_conversations(self, category):
        """
        Returns an iterator over full conversations in an ABCD category.
        :param category: str, ABCD category (e.g., 'train', 'dev', 'test').
        :return: iterator(string), iterator to conversations in the category.
        """
        for entry in self.abcd_data[category]:
            conversation = []
            for utterance_data in entry['original']:
                sender, utterance = utterance_data
                if sender == 'action':
                    continue
                if not utterance:
                    continue
                conversation.append(utterance)
            yield "\n".join(conversation)

    def get_sensitive_data(self, entry):
        """
        Extracts sensitive data from an entry of the ABCD dataset.
        :param entry: dict, ABCD entry.
        :return: dict, mapping of values that can be considered sensitive.
        """
        sensitive_data = {}
        personal_data = entry['scenario']['personal']
        order_data = entry['scenario']['order']
        for k in ['customer_name', 'username', 'email', 'phone', 'account_id']:
            value = personal_data.get(k)
            if value:
                sensitive_data[k] = value
        for k in ['order_id', 'street_address', 'zip_code']:
            value = order_data.get(k)
            if value:
                sensitive_data[k] = value
        return sensitive_data

    def get_sensitive_terms(self, category, tokenizer):
        """
        Extracts sensitive terms from the ABCD dataset.
        :param category: str, ABCD category (e.g., 'train', 'dev', 'test').
        :param tokenizer: tokenizer to split data in terms.
        :return: set, sensitive terms.
        """
        sensitive_terms = set()
        for entry in self.abcd_data[category]:
            for datum in self.get_sensitive_data(entry).values():
                for token in tokenizer.tokenize(datum):
                    sensitive_terms.add(token.term)
        return sensitive_terms

    def get_recurrent_data_regex(self, category):
        """
        Returns a regex that matches recurrent scenario data.
        :param category: str, ABCD category (e.g., 'train', 'dev', 'test').
        :return: regex matching the high frequency scenario values.
        """
        tokens = set()
        for entry in self.abcd_data[category]:
            personal_data = entry['scenario']['personal']
            for k in ['customer_name']:
                value = personal_data.get(k)
                if not value:
                    continue
                tokens.update([
                    re.escape(token)
                    for token in ABCD.whitespace_regex.split(value.lower())
                ])
        return re.compile('|'.join(tokens), re.I)

    def replace_unique_names(self):
        """
        Replaces names concatenating them the convo_id to create unique names.
        """
        for category, data in self.abcd_data.items():
            for entry in data:
                convo_id = str(entry['convo_id'])
                personal_data = entry['scenario']['personal']
                customer_name = personal_data['customer_name']
                customer_tokens = ABCD.whitespace_regex.split(customer_name)
                personal_data['customer_name'] = ' '.join(
                    [token + convo_id for token in customer_tokens]
                )
                tokens_regex = '|'.join(
                    [re.escape(token) for token in customer_tokens]
                )
                name_regex = re.compile(f"\\b{tokens_regex}\\b", re.I)
                replaced_original = []
                for utterance_data in entry['original']:
                    sender, utterance = utterance_data
                    replaced_utterance = ""
                    previous = 0
                    for match in name_regex.finditer(utterance):
                        end = match.end(0)
                        replaced_utterance += utterance[previous:end] + convo_id
                        previous = end
                    replaced_utterance += utterance[previous:]
                    replaced_original.append((sender, replaced_utterance))
                entry['original'] = replaced_original
                del entry['delexed']  # NOTE: Remove delexed for consistency.
