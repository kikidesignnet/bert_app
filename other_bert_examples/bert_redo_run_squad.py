import torch
import os
SQUAD_DIR='/Users/davidbressler/pythonstuff/bert_app/app/models'

os.chdir('/Users/davidbressler/pythonstuff/bert_app/app/pretrainedBERT/examples')

from pytorch_pretrained_bert import BertTokenizer, BertForQuestionAnswering
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import run_squad
import collections
import numpy as np
#from bs4 import BeautifulSoup
#from lxml import html
#import requests
import wikipedia


#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)

model_state_dict = torch.load('/Users/davidbressler/pythonstuff/bert_app/app/models/pytorch_model.bin', map_location='cpu')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased',state_dict=model_state_dict)
model.to(device) 





#dev-v1.1.json is the test dataset
predict_file='/Users/davidbressler/pythonstuff/squad_data/dev-v1.1.json'
#eval_examples is a list of 10570 'SquadExample' objects
eval_examples = run_squad.read_squad_examples(input_file=predict_file, is_training=False)
#each object contains fields for qas_id, question_text, and doc_tokens, etc.
print(dir(eval_examples[0]))

#For sake of speed, let's drastically reduce size of eval_examples
eval_examples=eval_examples[:100]

#eval_features is a list of 'run_squad.InputFeatures' objects
eval_features = run_squad.convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=384,
            doc_stride=128,
            max_query_length=64,
            is_training=False)
#each object has fields for tokens, input_mask, input_ids, segment_ids, etc.
#input_mask: I think all the examples have the same length (max_seq_length), so input_mask is just 1's (for good input) and 0's (right-padding)
#input_ids: numericalized tokens, then 0's (right-padding)
#segment_ids: 0's for query positions, 1's for document positions, then 0's (right-padding)
print(dir(eval_features[0]))
#tokens is a list of individual tokens
#[[CLS], query, [SEP], document, [SEP]]
#e.g. ['[CLS]', 'which', 'nfl', ..., '[SEP]', 'super', 'bowl', ..., '50', '.', '[SEP]']
print(eval_features[0].tokens)

#all_input_ids, all_input_mask, and all_segment_ids are Tensors w/ size([100, 384])
#all_example_index is just list w/ #s 0:99
all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)

# Run prediction for full data
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=8)

model.eval()
input_ids=all_input_ids[:5,:]
input_mask=all_input_mask[:5,:]
segment_ids=all_segment_ids[:5,:]
example_indices=all_example_index[:5]
input_ids = input_ids.to(device)
input_mask = input_mask.to(device)
segment_ids = segment_ids.to(device)

#batch_start_logits and batch_end_logits are both size [bs,384]
with torch.no_grad():
    batch_start_logits, batch_end_logits = model(input_ids, segment_ids, input_mask)

i=0
example_index=example_indices[0]
#start_logits and end_logits are both lists of len 384
start_logits = batch_start_logits[i].detach().cpu().tolist()
end_logits = batch_end_logits[i].detach().cpu().tolist()
#remember from above, eval_features is list of objects, each w/ fields for tokens, input_mask, input_ids, segment_ids, etc.
eval_feature = eval_features[example_index.item()] #here, just a single object
unique_id = int(eval_feature.unique_id)


#                start_logits = batch_start_logits[i].detach().cpu().tolist()
#                end_logits = batch_end_logits[i].detach().cpu().tolist()
#                eval_feature = eval_features[example_index.item()]
#                unique_id = int(eval_feature.unique_id)
#                all_results.append(RawResult(unique_id=unique_id,
#                                             start_logits=start_logits,
#                                             end_logits=end_logits))

#start_logits and end_logits get placed into "all_results"

#write_predictions(eval_examples, eval_features, all_results,
#                          args.n_best_size, args.max_answer_length,
#                          args.do_lower_case, output_prediction_file,
#                          output_nbest_file, args.verbose_logging)

#def write_predictions(all_examples, all_features, all_results, n_best_size,
#                      max_answer_length, do_lower_case, output_prediction_file,
#                      output_nbest_file, verbose_logging):

#n_best_size: the total number of n-best predictions to generate in the nbest_predictions.json
n_best_size=20
#both of these are lists (len(20)) of best guesses for start, and end, respectively
start_indexes = run_squad._get_best_indexes(start_logits, n_best_size)
end_indexes = run_squad._get_best_indexes(end_logits, n_best_size)

for start_index in start_indexes:
    for end_index in end_indexes:
        # We could hypothetically create invalid predictions, e.g., predict
        # that the start of the span is in the question. We throw out all
        # invalid predictions.
        if start_index >= len(feature.tokens):
            continue
        if end_index >= len(feature.tokens):
            continue
        if start_index not in feature.token_to_orig_map:
            continue
        if end_index not in feature.token_to_orig_map:
            continue
        if not feature.token_is_max_context.get(start_index, False):
            continue
        if end_index < start_index:
            continue
        length = end_index - start_index + 1
        if length > max_answer_length:
            continue
        prelim_predictions.append(
            _PrelimPrediction(
                feature_index=feature_index,
                start_index=start_index,
                end_index=end_index,
                start_logit=result.start_logits[start_index],
                end_logit=result.end_logits[end_index]))










#bert_app routes redo, with eval_examples approach:

#r = requests.get('https://en.wikipedia.org/wiki/Jimi_Hendrix')
#soup = BeautifulSoup(r.text, "lxml")
#soup = BeautifulSoup(r.text, "html.parser")
#document=soup.get_text().strip()

search_term="Jimmy Hendrix"
wik_page=wikipedia.search(search_term,results=1)
p = wikipedia.page(wik_page[0])
document = p.content # Content of page.

document='The University of Chicago (UChicago, Chicago, or U of C) is a private research university in Chicago. The university, established in 1890, consists of The College, various graduate programs, interdisciplinary committees organized into four academic research divisions and seven professional schools. Beyond the arts and sciences, Chicago is also well known for its professional schools, which include the Pritzker School of Medicine, the University of Chicago Booth School of Business, the Law School, the School of Social Service Administration, the Harris School of Public Policy Studies, the Graham School of Continuing Liberal and Professional Studies and the Divinity School. The university currently enrolls approximately 5,000 students in the College and around 15,000 students overall.'
query='What kind of university is the University of Chicago?'
#document='Nikola Tesla (Serbian Cyrillic: Никола Тесла; 10 July 1856 – 7 January 1943) was a Serbian American inventor, electrical engineer, mechanical engineer, physicist, and futurist best known for his contributions to the design of the modern alternating current (AC) electricity supply system.'
#query='Who was Tesla?'
#query='In what year was Nikola Tesla born?'
#query='What was Nikola Tesla s ethnicity?'
#query='In what year did Tesla die?'
#query='What does AC stand for?'

paragraph_text=document

def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

doc_tokens = []
char_to_word_offset = []
prev_is_whitespace = True
for c in paragraph_text:
    if is_whitespace(c):
        prev_is_whitespace = True
    else:
        if prev_is_whitespace:
            doc_tokens.append(c)
        else:
            doc_tokens[-1] += c
        prev_is_whitespace = False
    char_to_word_offset.append(len(doc_tokens) - 1)


#eval_examples is a list of 10570 'SquadExample' objects
#just using it as a format placeholder
#predict_file='/Users/davidbressler/pythonstuff/squad_data/dev-v1.1.json'
#eval_examples_routes = run_squad.read_squad_examples(input_file=predict_file, is_training=False)
#eval_examples_routes=[eval_examples_routes[0]]
#eval_examples_routes[0].question_text=query
#eval_examples_routes[0].doc_tokens=tokenizer.tokenize(document)

eval_examples_routes = [run_squad.SquadExample(
    qas_id=0,
    question_text=query,
    doc_tokens=doc_tokens,
    orig_answer_text=None,
    start_position=None,
    end_position=None)]

eval_features_routes = run_squad.convert_examples_to_features(
            examples=eval_examples_routes,
            tokenizer=tokenizer,
            max_seq_length=384,
            doc_stride=128,
            max_query_length=64,
            is_training=False)

#all_input_ids, all_input_mask, and all_segment_ids are Tensors w/ size([100, 384])
#all_example_index is just list w/ #s 0:99
all_input_ids_routes = torch.tensor([f.input_ids for f in eval_features_routes], dtype=torch.long)
all_input_mask_routes = torch.tensor([f.input_mask for f in eval_features_routes], dtype=torch.long)
all_segment_ids_routes = torch.tensor([f.segment_ids for f in eval_features_routes], dtype=torch.long)
all_example_index_routes = torch.arange(all_input_ids_routes.size(0), dtype=torch.long)
eval_data_routes = TensorDataset(all_input_ids_routes, all_input_mask_routes, all_segment_ids_routes, all_example_index_routes)

model.eval()
input_ids_routes=all_input_ids_routes
input_mask_routes=all_input_mask_routes
segment_ids_routes=all_segment_ids_routes
example_indices_routes=all_example_index_routes
input_ids_routes = input_ids_routes.to(device)
input_mask_routes = input_mask_routes.to(device)
segment_ids_routes = segment_ids_routes.to(device)

#batch_start_logits and batch_end_logits are both size [bs,384]
with torch.no_grad():
    batch_start_logits_routes, batch_end_logits_routes = model(input_ids_routes, segment_ids_routes, input_mask_routes)

i=0
example_index_routes=example_indices_routes[0]
#start_logits and end_logits are both lists of len 384
start_logits_routes = batch_start_logits_routes[i].detach().cpu().tolist()
end_logits_routes = batch_end_logits_routes[i].detach().cpu().tolist()
#remember from above, eval_features is list of objects, each w/ fields for tokens, input_mask, input_ids, segment_ids, etc.
eval_feature_routes = eval_features_routes[example_index_routes.item()] #here, just a single object
unique_id_routes = int(eval_feature_routes.unique_id)

start_ind=np.argmax(start_logits_routes)
end_ind=np.argmax(end_logits_routes)
the_answer=eval_feature_routes.tokens[start_ind:end_ind+1]
print(the_answer)

#n_best_size: the total number of n-best predictions to generate in the nbest_predictions.json
n_best_size=20
max_answer_length=30
#both of these are lists (len(20)) of best guesses for start, and end, respectively
start_indexes_routes = run_squad._get_best_indexes(start_logits_routes, n_best_size)
end_indexes_routes = run_squad._get_best_indexes(end_logits_routes, n_best_size)

_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "PrelimPrediction",
    ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

prelim_predictions = []
for start_index in start_indexes_routes:
    for end_index in end_indexes_routes:
        # We could hypothetically create invalid predictions, e.g., predict
        # that the start of the span is in the question. We throw out all
        # invalid predictions.
        if start_index >= len(eval_feature_routes.tokens):
            continue
        if end_index >= len(eval_feature_routes.tokens):
            continue
        if start_index not in eval_feature_routes.token_to_orig_map:
            continue
        if end_index not in eval_feature_routes.token_to_orig_map:
            continue
        if not eval_feature_routes.token_is_max_context.get(start_index, False):
            continue
        if end_index < start_index:
            continue
        length = end_index - start_index + 1
        if length > max_answer_length:
            continue
        prelim_predictions.append(
            _PrelimPrediction(
                feature_index=0,
                start_index=start_index,
                end_index=end_index,
                start_logit=start_logits_routes[start_index],
                end_logit=end_logits_routes[end_index]))

#prelim_predictions is a list of PrelimPrediction's
#example: PrelimPrediction(feature_index=0, start_index=30, end_index=31, start_logit=7.1346, end_logit=5.40855)
prelim_predictions = sorted(
    prelim_predictions,
    key=lambda x: (x.start_logit + x.end_logit),
    reverse=True)

_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "NbestPrediction", ["text", "start_logit", "end_logit"])

seen_predictions = {}
nbest = []
for pred in prelim_predictions:
    if len(nbest) >= n_best_size:
        break
    tok_tokens = eval_feature_routes.tokens[pred.start_index:(pred.end_index + 1)]
    orig_doc_start = eval_feature_routes.token_to_orig_map[pred.start_index]
    orig_doc_end = eval_feature_routes.token_to_orig_map[pred.end_index]
    orig_tokens = doc_tokens[orig_doc_start:(orig_doc_end + 1)]
    tok_text = " ".join(tok_tokens)
    # De-tokenize WordPieces that have been split off.
    tok_text = tok_text.replace(" ##", "")
    tok_text = tok_text.replace("##", "")
    # Clean whitespace
    tok_text = tok_text.strip()
    tok_text = " ".join(tok_text.split())
    orig_text = " ".join(orig_tokens)
    final_text = run_squad.get_final_text(tok_text, orig_text, do_lower_case=True, verbose_logging=False)
    if final_text in seen_predictions:
        continue
    seen_predictions[final_text] = True
    nbest.append(
        _NbestPrediction(
            text=final_text,
            start_logit=pred.start_logit,
            end_logit=pred.end_logit))
    # In very rare edge cases we could have no valid predictions. So we
    # just create a nonce prediction in this case to avoid failure.
    if not nbest:
        nbest.append(
            _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))
    assert len(nbest) >= 1
    total_scores = []
    for entry in nbest:
        total_scores.append(entry.start_logit + entry.end_logit)
    probs = run_squad._compute_softmax(total_scores)

print(nbest[0].text)





#bert_app routes redo:


document='The University of Chicago (UChicago, Chicago, or U of C) is a private research university in Chicago. The university, established in 1890, consists of The College, various graduate programs, interdisciplinary committees organized into four academic research divisions and seven professional schools. Beyond the arts and sciences, Chicago is also well known for its professional schools, which include the Pritzker School of Medicine, the University of Chicago Booth School of Business, the Law School, the School of Social Service Administration, the Harris School of Public Policy Studies, the Graham School of Continuing Liberal and Professional Studies and the Divinity School. The university currently enrolls approximately 5,000 students in the College and around 15,000 students overall.'
query='What kind of university is the University of Chicago?'

#tokenized input
document_tokens = tokenizer.tokenize(document)
query_tokens = tokenizer.tokenize(query)
all_tokens= ['[CLS]'] + query_tokens + ['[SEP]']  + document_tokens + ['[SEP]']
# Convert tokens to vocabulary indices
all_indices= tokenizer.convert_tokens_to_ids(all_tokens)
# Define sentence A and B indices associated to 1st and 2nd sentences
query_segids = [0 for i in range(len(query_tokens) +1 )]
document_segids = [1 for i in range(len(document_tokens) +2 )]
all_segids= query_segids + document_segids
assert len(all_segids) == len(all_indices)
# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([all_indices])
segments_tensors = torch.tensor([all_segids])
input_mask_routes = torch.tensor([1 for i in range(len(all_segids))])
input_mask_routes=input_mask_routes.view(1,-1)
# Predict all tokens
with torch.no_grad(): #https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/8
    start_logits_routes, end_logits_routes = model(tokens_tensor, segments_tensors,input_mask_routes)

start_logits_routes=start_logits_routes.cpu().tolist()[0]
end_logits_routes=end_logits_routes.cpu().tolist()[0]

n_best_size=20 #number of guesses allowed, per start and end
max_answer_length=30
#both of these are lists (len(20)) of best guesses for start, and end, respectively
start_indexes_routes = run_squad._get_best_indexes(start_logits_routes, n_best_size)
end_indexes_routes = run_squad._get_best_indexes(end_logits_routes, n_best_size)

_PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "PrelimPrediction",
    ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

prelim_predictions = []
for start_index in start_indexes_routes:
    for end_index in end_indexes_routes:
        # We could hypothetically create invalid predictions, e.g., predict
        # that the start of the span is in the question. We throw out all
        # invalid predictions.
        if start_index >= len(all_tokens):
            continue
        if end_index >= len(all_tokens):
            continue
        #if start_index not in feature.token_to_orig_map:
        #    continue
        #if end_index not in feature.token_to_orig_map:
        #    continue
        #if not feature.token_is_max_context.get(start_index, False):
        #    continue
        if end_index < start_index:
            continue
        length = end_index - start_index + 1
        if length > max_answer_length:
            continue
        prelim_predictions.append(
            _PrelimPrediction(
                feature_index=0,
                start_index=start_index,
                end_index=end_index,
                start_logit=start_logits_routes[start_index],
                end_logit=end_logits_routes[end_index]))

#prelim_predictions is a list of PrelimPrediction's
#example: PrelimPrediction(feature_index=0, start_index=30, end_index=31, start_logit=7.1346, end_logit=5.40855)
prelim_predictions = sorted(
    prelim_predictions,
    key=lambda x: (x.start_logit + x.end_logit),
    reverse=True)

_NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
    "NbestPrediction", ["text", "start_logit", "end_logit"])

seen_predictions = {}
nbest = []
for pred in prelim_predictions:
    if len(nbest) >= n_best_size:
        break
    tok_tokens = all_tokens[pred.start_index:(pred.end_index + 1)]
    orig_doc_start = feature.token_to_orig_map[pred.start_index]
    orig_doc_end = feature.token_to_orig_map[pred.end_index]
    orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
    tok_text = " ".join(tok_tokens)





