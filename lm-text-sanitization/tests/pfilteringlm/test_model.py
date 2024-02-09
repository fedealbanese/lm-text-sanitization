from asapp.pfilteringlm import ULM, FillMaskModel, GPT2Model, T5Model


def test_model_doc_repr():
    phrase = "two words"
    models = {
        "ulm": ULM(),
        "bert": FillMaskModel(model_name = 'bert-base-uncased'),
        "roberta": FillMaskModel(model_name = 'roberta-base'),
        "albert": FillMaskModel(model_name = 'albert-base-v2'),
        "gpt2": GPT2Model(model_name = 'gpt2'),
        "distilgpt2":GPT2Model(model_name = 'distilgpt2'),
        "t5": T5Model(model_name = 't5-small'),
        }
    for model_type in models.keys():
        if model_type == "ulm":
            models[model_type].add_doc(phrase)
        token_doc_representation = models[model_type].get_doc_repr(phrase)
        assert len(token_doc_representation) == 2
        assert token_doc_representation[0].term == "two"
        assert token_doc_representation[0].start == 0
        assert token_doc_representation[0].end == 3
        assert token_doc_representation[1].term == "words"
        assert token_doc_representation[1].start == 4
        assert token_doc_representation[1].end == 9
