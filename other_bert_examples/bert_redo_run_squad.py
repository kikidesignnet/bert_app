import torch
import os
SQUAD_DIR='/Users/davidbressler/pythonstuff/bert_app/app/models'

os.chdir('/Users/davidbressler/pythonstuff/pytorch-pretrained-BERT/examples')

from pytorch_pretrained_bert import BertTokenizer, BertForQuestionAnswering
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import run_squad

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)

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
#[CLS], query, [SEP], document, [SEP]
#e.g. ['[CLS]', 'which', 'nfl', ..., '[SEP]', 'super', 'bowl', ..., '50', '.', '[SEP]']
print(eval_features[0].tokens)

#all_input_ids, all_input_mask, and all_segment_ids are Size([100, 384])
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


#write_predictions(eval_examples, eval_features, all_results,
#                          args.n_best_size, args.max_answer_length,
#                          args.do_lower_case, output_prediction_file,
#                          output_nbest_file, args.verbose_logging)