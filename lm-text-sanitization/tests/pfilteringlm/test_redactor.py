from asapp.pfilteringlm import ULM, FillMaskModel, GPT2Model, T5Model
from asapp.commons import Redactor
 

def test_redactor():
    test_document = "My phone number is 44444444"
    models = {
        "ulm": {"model": ULM(), "p_threshold": 1e-3},
        "bert": {"model": FillMaskModel(model_name = 'bert-base-uncased'), "p_threshold": 1e-3},
        "roberta": {"model": FillMaskModel(model_name = 'roberta-base'),  "p_threshold": 1e-6},
        "albert": {"model": FillMaskModel(model_name = 'albert-base-v2'),  "p_threshold": 1e-7},
        "gpt2": {"model": GPT2Model(model_name = 'gpt2'),  "p_threshold": 1e-7},
        "distilgpt2": {"model": GPT2Model(model_name = 'distilgpt2'),  "p_threshold": 1e-7},
        "t5": {"model": T5Model(model_name = 't5-small'),  "p_threshold": 1e-7},
        }
    for model_type in models.keys():
        if model_type == "ulm":
            models[model_type]["model"].add_doc("My phone number is")
        redactor = Redactor(
            models[model_type]["model"], 
            p = models[model_type]["p_threshold"]
            )
        redacted_document = redactor.redact(test_document)
        redacted_unsafe = redactor.redact_term("44444444")
        expected = "My phone number is " + redacted_unsafe
        assert redacted_document == expected
