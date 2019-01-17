import torch
import os
SQUAD_DIR='/Users/davidbressler/pythonstuff/bert_app/app/models'

os.chdir('/Users/davidbressler/pythonstuff/pytorch-pretrained-BERT/examples')

from pytorch_pretrained_bert import BertTokenizer, BertForQuestionAnswering
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import run_squad

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model (weights)
model_state_dict = torch.load('/data/squad/pytorch_model.bin')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased',state_dict=model_state_dict)
model.to(device) #DO I NEED TO DO ANYTHING WITH THIS?
model.eval()

#inputs
#document='Born in Seattle, Washington, Hendrix began playing guitar at the age of 15.'
#query='What did Hendrix play?'
#query='Where was Hendrix born?'
#query='How old was Hendrix when he began playing guitar?'
#query='How old was Hendrix when he began playing music?'
#query='Where is the birthplace of Hendrix?'
document='The University of Chicago (UChicago, Chicago, or U of C) is a private research university in Chicago. The university, established in 1890, consists of The College, various graduate programs, interdisciplinary committees organized into four academic research divisions and seven professional schools. Beyond the arts and sciences, Chicago is also well known for its professional schools, which include the Pritzker School of Medicine, the University of Chicago Booth School of Business, the Law School, the School of Social Service Administration, the Harris School of Public Policy Studies, the Graham School of Continuing Liberal and Professional Studies and the Divinity School. The university currently enrolls approximately 5,000 students in the College and around 15,000 students overall.'
query='What kind of university is the University of Chicago?'
#document='Nikola Tesla (Serbian Cyrillic: Никола Тесла; 10 July 1856 – 7 January 1943) was a Serbian American inventor, electrical engineer, mechanical engineer, physicist, and futurist best known for his contributions to the design of the modern alternating current (AC) electricity supply system.'
#query='In what year was Nikola Tesla born?'
#query='What was Nikola Tesla s ethnicity?'
#query='In what year did Tesla die?'
#query='What does AC stand for?'

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
tokens_tensor = torch.tensor([all_indices]).cuda()
segments_tensors = torch.tensor([all_segids]).cuda()
input_mask = torch.tensor([1 for i in range(len(all_segids))]).cuda()
input_mask=input_mask.view(1,-1)

# Predict all tokens
start_logits, end_logits = model(tokens_tensor, segments_tensors,input_mask)
start_ind=torch.argmax(start_logits).item()
end_ind=torch.argmax(end_logits).item()

print(all_tokens[start_ind:end_ind+1])




#
#Messing around, trying to recreate what happened in run_squad.py

predict_file='/data/squad/dev-v1.1.json'
#eval_examples is a list of 10570 'SquadExample' objects
#each object contains fields for qas_id, question_text, and doc_tokens, 
eval_examples = run_squad.read_squad_examples(input_file=predict_file, is_training=False)

eval_features = run_squad.convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=384,
            doc_stride=128,
            max_query_length=64,
            is_training=False)

#write_predictions(eval_examples, eval_features, all_results,
#                          args.n_best_size, args.max_answer_length,
#                          args.do_lower_case, output_prediction_file,
#                          output_nbest_file, args.verbose_logging)