from flask import render_template, flash, redirect
from app import app
from app.forms import QueryForm
from pytorch_pretrained_bert import BertTokenizer, BertForQuestionAnswering
from app.pretrainedBERT.examples import run_squad
from torch.utils.data import TensorDataset
#from bs4 import BeautifulSoup

#CHANGES: This step produces warning "Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex."

#CHANGES: WARNING: Do not use the development server in a production environment. Use a production WSGI server instead.

#import pytorch_pretrained_bert.examples.run_squad

import torch
import os
import collections
import wikipedia



# Load pre-trained model tokenizer (vocabulary)
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',do_lower_case=True)
#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model (weights)
filename=os.path.join(app.root_path, 'models', 'pytorch_model.bin')
model_state_dict = torch.load(filename, map_location='cpu')
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased',state_dict=model_state_dict)
model.to(device) 
model.eval()


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    print(app.root_path)
    form= QueryForm()
    if form.validate_on_submit():
        
        search_term="Jimmy Hendrix"
        wik_page=wikipedia.search(search_term,results=1)
        p = wikipedia.page(wik_page[0])
        paragraph_text = p.content # Content of page.
        wik_url=p.url

        query=form.the_query.data
        paragraph_text=form.the_document.data

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

        the_answer=nbest[0].text



        # #tokenized input
        # document_tokens = tokenizer.tokenize(form.the_document.data)
        # query_tokens = tokenizer.tokenize(form.the_query.data)
        # all_tokens= ['[CLS]'] + query_tokens + ['[SEP]']  + document_tokens + ['[SEP]']
        # # Convert tokens to vocabulary indices
        # all_indices= tokenizer.convert_tokens_to_ids(all_tokens)
        # # Define sentence A and B indices associated to 1st and 2nd sentences
        # query_segids = [0 for i in range(len(query_tokens) +1 )]
        # document_segids = [1 for i in range(len(document_tokens) +2 )]
        # all_segids= query_segids + document_segids
        # assert len(all_segids) == len(all_indices)
        # # Convert inputs to PyTorch tensors
        # tokens_tensor = torch.tensor([all_indices])
        # segments_tensors = torch.tensor([all_segids])
        # input_mask = torch.tensor([1 for i in range(len(all_segids))])
        # input_mask=input_mask.view(1,-1)
        # # Predict all tokens
        # with torch.no_grad(): #https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/8
        #     start_logits, end_logits = model(tokens_tensor, segments_tensors,input_mask)
        # start_ind=torch.argmax(start_logits).item()
        # end_ind=torch.argmax(end_logits).item()
        #the_answer=all_tokens[start_ind:end_ind+1]

        return render_template('index.html',title='Home', form=form, wik_url=wik_url, the_wik_search=form.the_wik_search.data, the_document=form.the_document.data, the_query=form.the_query.data, the_answer=the_answer)

        #flash('Your Query: {}'.format(
        #    form.the_query.data))
        #flash('The Document: {}'.format(
        #    form.the_document.data))
        #return redirect('/index')
    return render_template('index.html',title='Home', form=form, wik_url=None, the_wik_search=None, the_document=None, the_query=None, the_answer=None)


