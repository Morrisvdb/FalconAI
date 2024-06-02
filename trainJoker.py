from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np, os, csv, json, torch

from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


import logging
logging.getLogger().setLevel(logging.CRITICAL)

import warnings
warnings.filterwarnings('ignore')

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

BATCH_SIZE = 4
EPOCHS = 25
LEARNING_RATE = 3e-5
WARMUP_STEPS = 5000
MAX_SEQ_LEN = 400

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model = model.to(device)

def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)

class JokesDataset(Dataset):
    def __init__(self, jokes_dataset_path = 'data/'):
        super().__init__()

        short_jokes_path = os.path.join(jokes_dataset_path, 'shortjokes.csv')

        self.joke_list = []
        self.end_of_text_token = "<|endoftext|>"
        
        with open(short_jokes_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            
            x = 0
            for row in csv_reader:
                joke_str = f"JOKE:{row[1]}{self.end_of_text_token}"
                self.joke_list.append(joke_str)
        
    def __len__(self):
        return len(self.joke_list)

    def __getitem__(self, item):
        return self.joke_list[item]

dataset = JokesDataset()
joke_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

model = model.to(device)
model.train()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=-1)
proc_seq_count = 0
sum_loss = 0.0
batch_count = 0

tmp_jokes_tens = None
models_folder = "joker_models"
if not os.path.exists(models_folder):
    os.mkdir(models_folder)

for epoch in range(EPOCHS):
    
    print(f"EPOCH {epoch} started" + '=' * 30)
    
    from tqdm import tqdm
    progress_bar = tqdm(enumerate(joke_loader), total=len(joke_loader))
    for idx,joke in progress_bar:
        
        #################### "Fit as many joke sequences into MAX_SEQ_LEN sequence as possible" logic start ####
        joke_tens = torch.tensor(tokenizer.encode(joke[0])).unsqueeze(0).to(device)
        #Skip sample from dataset if it is longer than MAX_SEQ_LEN
        if joke_tens.size()[1] > MAX_SEQ_LEN:
            continue
        
        #The first joke sequence in the sequence
        if not torch.is_tensor(tmp_jokes_tens):
            tmp_jokes_tens = joke_tens
            continue
        else:
            #The next joke does not fit in so we process the sequence and leave the last joke 
            #as the start for next sequence 
            if tmp_jokes_tens.size()[1] + joke_tens.size()[1] > MAX_SEQ_LEN:
                work_jokes_tens = tmp_jokes_tens
                tmp_jokes_tens = joke_tens
            else:
                #Add the joke to sequence, continue and try to add more
                tmp_jokes_tens = torch.cat([tmp_jokes_tens, joke_tens[:,1:]], dim=1)
                continue
        ################## Sequence ready, process it trough the model ##################
            
        outputs = model(work_jokes_tens, labels=work_jokes_tens)
        loss, logits = outputs[:2]                        
        loss.backward()
        sum_loss = sum_loss + loss.detach().data
                       
        proc_seq_count = proc_seq_count + 1
        if proc_seq_count == BATCH_SIZE:
            proc_seq_count = 0    
            batch_count += 1
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.zero_grad()

        if batch_count == 100:
            print(f"sum loss {sum_loss}")
            batch_count = 0
            sum_loss = 0.0
    

    model.save_pretrained(models_folder)
