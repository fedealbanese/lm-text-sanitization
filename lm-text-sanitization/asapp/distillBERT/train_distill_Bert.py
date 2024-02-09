#------------------
#-------MODEL------
#------------------
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertModel, BertEmbeddings, BertOnlyMLMHead
from torch.nn import Module
from collections import OrderedDict
import torch
import math
import os

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
   size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

def narrow_tensor(key, teacher_state_dict_key, new_embedding_size, old_embedding_size, old_intermediate_size):
    #For embedding size
    indices = [i for i, x in enumerate(list(teacher_state_dict_key.shape)) if x == old_embedding_size]
    return_tensor = teacher_state_dict_key
    for index in indices:
        return_tensor = torch.narrow(
                          input=return_tensor,
                          dim=index,
                          start=0,
                          length=new_embedding_size,
                          )
    #For intermediate size
    
    indices = [i for i, x in enumerate(list(teacher_state_dict_key.shape)) if x == old_intermediate_size]
    for index in indices:
        return_tensor = torch.narrow(
                          input=return_tensor,
                          dim=index,
                          start=0,
                          length=int(new_embedding_size*(old_intermediate_size/old_embedding_size)),
                          )
    
    return return_tensor

def distill_bert_weights(
    teacher : Module,
    student : Module,
    n_layers : int  = 2,
    new_embedding_size : int  = 768,
    old_embedding_size : int  = 768,
    old_intermediate_size: int = 3072,
) -> None:
    """
    Recursively copies the weights of the (teacher) to the (student).
    This function is meant to be first called on a BertFor... model, but is then called on every children of that model recursively.
    """
    # If the part is an entire BERT model or a BertaFor..., unpack and iterate
    if isinstance(teacher, BertModel) or type(teacher).__name__.startswith('BertFor'):
        for teacher_part, student_part in zip(teacher.children(), student.children()):
            distill_bert_weights(teacher_part, student_part, n_layers, new_embedding_size, old_embedding_size)
    # Else if the part is an encoder, copy one out of every layer
    elif isinstance(teacher, BertEncoder):
            teacher_encoding_layers = [layer for layer in next(teacher.children())]
            student_encoding_layers = [layer for layer in next(student.children())]
            for i in range(len(student_encoding_layers)):
                student_state_dict = OrderedDict()
                for key in teacher_encoding_layers[n_layers*i].state_dict().keys():
                      student_state_dict[key] = narrow_tensor(
                          key,
                          teacher_encoding_layers[n_layers*i].state_dict()[key],
                          new_embedding_size,
                          old_embedding_size,
                          old_intermediate_size
                          )
                student_encoding_layers[i].load_state_dict(student_state_dict)

    elif isinstance(teacher, BertEmbeddings) or isinstance(teacher, BertOnlyMLMHead):
            student_state_dict = OrderedDict()
            for key in teacher.state_dict().keys():
                  student_state_dict[key] = narrow_tensor(
                      key,
                      teacher.state_dict()[key],
                      new_embedding_size,
                      old_embedding_size,
                      old_intermediate_size
                      )
            student.load_state_dict(student_state_dict)

    # Else the part is a head or something else, copy the state_dict
    else:
        print(teacher.state_dict().keys())
        student.load_state_dict(teacher.state_dict())

def distill_bert(
    teacher_model : BertPreTrainedModel, n_layers : int  = 12, new_embedding_size : int  = 768,
) -> BertPreTrainedModel:
    """
    Distilates a Bert model (teacher_model).
    The student model has the same configuration, except for the number of hidden layers, which is // by n_layers and the embedding size, which is new_embedding_size.
    The student layers are initilized by copying and narrow the layers of the teacher, starting with layer 0.
    The head of the teacher is also copied.
    """
    # Get teacher configuration as a dictionnary
    configuration = teacher_model.config.to_dict()
    # Half the number of hidden layer
    configuration['num_hidden_layers'] //= n_layers
    #reduce embedding size MIO
    old_embedding_size = configuration['hidden_size']
    old_intermediate_size = configuration['intermediate_size']
    configuration['hidden_size'] = new_embedding_size 
    configuration['intermediate_size'] = int(new_embedding_size*(old_intermediate_size/old_embedding_size))
    # Convert the dictionnary to the student configuration
    configuration = BertConfig.from_dict(configuration)
    # Create uninitialized student model
    student_model = type(teacher_model)(configuration)
    # Initialize the student's weights
    distill_bert_weights(teacher=teacher_model, student=student_model, n_layers=n_layers, new_embedding_size=new_embedding_size, old_embedding_size=old_embedding_size, old_intermediate_size=old_intermediate_size)
    # Return the student model
    return student_model


device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_checkpoint = "amine/bert-base-5lang-cased"
model_teacher_bert = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
n_layers = 6
hidden_size = 120
model_student = distill_bert(model_teacher_bert, n_layers, hidden_size)

print("End distillation")

#------------------
#-------Evaluate---
#------------------

#Resultados pretraining
text = "My name is [MASK] Smith."
index_word = 4 
top_n = 10
results = model_student(torch.tensor([tokenizer(text)["input_ids"]]).to(device)) #https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForMaskedLM.forward
ps = results.logits[0,index_word,:] 
top_k = torch.topk(ps,top_n)
for v, i in zip(top_k.values, top_k.indices):
  print(v.item(), i.item(), tokenizer.decode(i.item()))

#Resultados pretraining
text = "Mi nombre es [MASK] Perez."
index_word = 4
top_n = 10
results = model_student(torch.tensor([tokenizer(text)["input_ids"]]).to(device)) #https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForMaskedLM.forward
ps = results.logits[0,index_word,:] 
top_k = torch.topk(ps,top_n)
for v, i in zip(top_k.values, top_k.indices):
  print(v.item(), i.item(), tokenizer.decode(i.item()))

#------------------
#-------Data-------
#------------------
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from transformers import default_data_collator
from datasets import interleave_datasets

def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

subsamples = 1_000_000
chunk_size = 128 #Lower than: print(tokenizer.model_max_length)
train_size = 900_000
test_size = int(0.1 * train_size)
batch_size = 256

wiki_es = load_dataset('large_spanish_corpus', name='all_wikis', streaming=True)
wiki_en = load_dataset("wikipedia", "20220301.en", streaming=True)
wiki_es["train"] = wiki_es["train"].shuffle(seed=42).take(subsamples)
wiki_en["train"] = wiki_en["train"].shuffle(seed=42).take(subsamples)
wiki_en = wiki_en.remove_columns(['id', 'url', 'title'])
all_wiki = interleave_datasets([wiki_en['train'], wiki_es['train']])
all_wiki = all_wiki.shuffle(seed=42).take(subsamples)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
tokenized_datasets = all_wiki.map(tokenize_function, batched=True, remove_columns=["text"] ) # Use batched=True to activate fast multithreading!

lm_datasets = tokenized_datasets.map(group_texts, batched=True)
downsampled_dataset_train = lm_datasets.take(train_size).remove_columns(["word_ids"])
downsampled_dataset_test = lm_datasets.skip(train_size).take(test_size).remove_columns(["word_ids"])
eval_dataset = downsampled_dataset_test.map(
    insert_random_mask,
    batched=True,
    remove_columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
)
eval_dataset = eval_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    }
)
eval_dataset = eval_dataset.remove_columns(['masked_token_type_ids'])

train_dataloader = DataLoader(downsampled_dataset_train, batch_size=batch_size, collate_fn=data_collator)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=default_data_collator)

print("End dataset loading and preprocessing")

#------------------
#-------Loss-------
#------------------
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss

def distillation_loss(
    teacher_logits,
    student_logits,
    labels,
    temperature : float = 1.0,
):
    """
    The distillation loss for distilating a BERT-like model.
    The loss takes the (teacher_logits), (student_logits) and (labels) for various losses.
    The (temperature) can be given, otherwise it's set to 1 by default.
    """
    # Temperature and sotfmax
    student_logits = torch.softmax(student_logits,dim = 1)
    teacher_logits = torch.softmax(teacher_logits,dim = 1)
    # Classification loss (problem-specific loss)
    loss = CrossEntropyLoss()(
        torch.transpose(student_logits, 1, 2), 
        labels
        )
    # CrossEntropy teacher-student loss
    loss = loss + CrossEntropyLoss()(
        torch.transpose(student_logits, 1, 2), 
        torch.transpose(teacher_logits, 1, 2)
        )
    # Cosine loss
    size = student_logits.size()
    loss = loss + CosineEmbeddingLoss()(
        torch.reshape(teacher_logits, (size[0]*size[1], size[2])), 
        torch.reshape(student_logits, (size[0]*size[1], size[2])),
        torch.ones(size[0]*size[1])
        )
    # Average the loss and return it
    loss = loss / 3
    return loss

#------------------
#-------Train------
#------------------

from torch.optim import AdamW
from transformers import get_scheduler
from accelerate import Accelerator
from huggingface_hub import get_full_repo_name
from tqdm.auto import tqdm

model_name = "" #Complete
repo_name = get_full_repo_name(model_name, token="") #Complete

num_train_epochs = 4
optimizer = AdamW(model_student.parameters(), lr=1e-4)
accelerator = Accelerator()
model_student, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model_student, optimizer, train_dataloader, eval_dataloader
)
num_update_steps_per_epoch = train_size 
num_training_steps = num_train_epochs * num_update_steps_per_epoch
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model_student.train()
    for batch in train_dataloader:
        teacher_outputs = model_teacher_bert(**batch)
        outputs = model_student(**batch)
        loss =  distillation_loss(
            teacher_logits = teacher_outputs.logits,
            student_logits = outputs.logits,
            labels = batch["labels"],
            )
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model_student.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model_student(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(batch_size)))

    losses = torch.cat(losses)
    losses = losses[: test_size]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity} Loss:{losses[-1]}")
    model_student.save_pretrained(f"model_student_epoch{epoch}")

print("End training")

#------------------
#-------Evaluate---
#------------------

#Resultados pretraining
text = "My name is [MASK] Smith."
index_word = 4
top_n = 10
results = model_student(torch.tensor([tokenizer(text)["input_ids"]]).to(device)) #https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForMaskedLM.forward
ps = results.logits[0,index_word,:] 
top_k = torch.topk(ps,top_n)
for v, i in zip(top_k.values, top_k.indices):
  print(v.item(), i.item(), tokenizer.decode(i.item()))

#Resultados pretraining
text = "Mi nombre es [MASK] Perez."
index_word = 4 
top_n = 10
results = model_student(torch.tensor([tokenizer(text)["input_ids"]]).to(device)) #https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForMaskedLM.forward
ps = results.logits[0,index_word,:] 
top_k = torch.topk(ps,top_n)
for v, i in zip(top_k.values, top_k.indices):
  print(v.item(), i.item(), tokenizer.decode(i.item()))


#------------------
#-------Save-------
#------------------

model_student.save_pretrained("model_student")
path = "model_student/pytorch_model.bin"
convert_size(os.path.getsize(path))
