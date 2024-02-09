
from asapp.kanonymity import KAnonymizer
from asapp.commons import Redactor


def test_redact():
    kanonymizer = KAnonymizer(k=2)
    kanonymizer.add_doc("Hola Don Pepito")
    kanonymizer.add_doc("Hola Don Jose")
    redactor = Redactor(kanonymizer, p=0.5)
    redaction = redactor.redact("Hola Don Fede")
    assert redaction == f"Hola Don {redactor.redact_term('Fede')}"
