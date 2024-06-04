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

BATCH_SIZE = 2
EPOCHS = 25
LEARNING_RATE = 3e-5
WARMUP_STEPS = 5000
MAX_SEQ_LEN = 400

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model = model.to(device)

def generate_jokes(count, model):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    joke_num = 0
    output = []
    with torch.no_grad():
   
        for joke_idx in range(count):
        
            joke_finished = False

            cur_ids = torch.tensor(tokenizer.encode("JOKE:")).unsqueeze(0).to(device)

            for i in range(100):
                outputs = model(cur_ids, labels=cur_ids)
                loss, logits = outputs[:2]
                softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(from only one in this case) batch and the last predicted embedding
                if i < 3:
                    n = 20
                else:
                    n = 3
                next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=n) #Randomly(from the topN probability distribution) select the next word
                cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1) # Add the last word to the running sequence

                if next_token_id in tokenizer.encode('<|endoftext|>'):
                    joke_finished = True
                    break

            
            if joke_finished:
                
                joke_num = joke_num + 1
                
                output_list = list(cur_ids.squeeze().to('cpu').numpy())
                output_text = tokenizer.decode(output_list)

                output.append(output_text.split("<|endoftext|>")[0])
    return output


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
        self.end_of_text_token = "<||>"
        
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
    
class ValJokesDataset(Dataset):
    def __init__(self, jokes_dataset_path = 'data/', fraction=0.05):
        super().__init__()

        short_jokes_path = os.path.join(jokes_dataset_path, 'shortjokes.csv')

        self.joke_list = []
        self.end_of_text_token = "<|endoftext|>"
        
        with open(short_jokes_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')            

            rows = []
            for row in csv_reader:
                rows.append(row)


            for row in rows[-int(len(rows)*fraction):]:
                joke_str = f"JOKE:{row[1]}{self.end_of_text_token}"
                self.joke_list.append(joke_str)
        
    def __len__(self):
        return len(self.joke_list)

    def __getitem__(self, item):
        return self.joke_list[item]


dataset = JokesDataset()
joke_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataset = ValJokesDataset()
def collate_fn(batch):
    jokes = [item[0] for item in batch]
    encoded = tokenizer.batch_encode_plus(jokes, padding=True, return_tensors='pt')
    return encoded['input_ids'].to(device), encoded['attention_mask'].to(device)

validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
print(len(dataset))
print(len(validation_dataset))


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
        average_loss = sum_loss / (idx + 1)
                       
        proc_seq_count = proc_seq_count + 1
        if proc_seq_count == BATCH_SIZE:
            proc_seq_count = 0    
            batch_count += 1
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model.zero_grad()

        if batch_count == 100:

            # model.eval()
            # with torch.no_grad():  # Do not calculate gradients to save memory
            #     val_sum_loss = 0.0
            #     val_batch_count = 0
            #     val_bar = tqdm(enumerate(validation_loader), total=len(validation_loader))
            #     val_bar.set_description(f"Validation loss")
            #     for idx, joke in val_bar:
            #         joke_tens = joke[0].to(device)
            #         # joke_tens = torch.tensor(tokenizer.encode(joke[0])).unsqueeze(0).to(device)
            #         if joke_tens.size()[1] > MAX_SEQ_LEN:
            #             continue
            #         outputs = model(joke_tens, labels=joke_tens)
            #         loss, logits = outputs[:2]
            #         val_sum_loss = val_sum_loss + loss.detach().data
            #         val_batch_count += 1
            #     val_average_loss = val_sum_loss / val_batch_count
            #     print(f"Validation average loss {val_average_loss}")
            # model.train()
            print(generate_jokes(model=model, count=5))
            print(f"sum loss {sum_loss}")
            print(f"average loss {average_loss}")
            batch_count = 0
            sum_loss = 0.0

    
    

    model.save_pretrained(models_folder)
