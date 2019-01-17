from flask import render_template, flash, redirect
from app import app
from app.forms import QueryForm
from pytorch_pretrained_bert import BertTokenizer, BertForQuestionAnswering
#CHANGES: This step produces warning "Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex."

#CHANGES: WARNING: Do not use the development server in a production environment. Use a production WSGI server instead.

#from app.pretrainedBERT.examples import run_squad
#import pytorch_pretrained_bert.examples.run_squad

import torch
import os

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
        
        #tokenized input
        document_tokens = tokenizer.tokenize(form.the_document.data)
        query_tokens = tokenizer.tokenize(form.the_query.data)
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
        input_mask = torch.tensor([1 for i in range(len(all_segids))])
        input_mask=input_mask.view(1,-1)
        # Predict all tokens
        with torch.no_grad(): #https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/8
            start_logits, end_logits = model(tokens_tensor, segments_tensors,input_mask)
        start_ind=torch.argmax(start_logits).item()
        end_ind=torch.argmax(end_logits).item()
        the_answer=all_tokens[start_ind:end_ind+1]

        return render_template('index.html',title='Home', form=form, the_document=form.the_document.data, the_query=form.the_query.data, the_answer=the_answer)

        #flash('Your Query: {}'.format(
        #    form.the_query.data))
        #flash('The Document: {}'.format(
        #    form.the_document.data))
        #return redirect('/index')
    return render_template('index.html',title='Home', form=form, the_document=None, the_query=None, the_answer=None)


