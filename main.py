import torch, tqdm, csv, json, os, numpy as np, math, logging, uuid, warnings
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from generate import generate

def get_unique_model_name():
    return str(uuid.uuid4())

BATCH_SIZE = 16
BATCH_SIZE = input(f"Enter batch size ({BATCH_SIZE}): ") or BATCH_SIZE
try:
    BATCH_SIZE = int(BATCH_SIZE)
except ValueError:
    print("Invalid batch size, aborting...")
    exit()

EPOCHS = 25
EPOCHS = input(f"Enter number of epochs ({EPOCHS}): ") or EPOCHS
try:
    EPOCHS = int(EPOCHS)
except ValueError:
    print("Invalid number of epochs, aborting...")
    exit()
    
LEARNING_RATE = 5e-6
WARMUP_STEPS = 5000
MAX_SEQ_LEN = 400
MAX_LR = 1e-3

UNIQUE_MODEL_NAME = get_unique_model_name()


logging.getLogger().setLevel(logging.CRITICAL)

warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training model on: ", device)

mode = input("Choose input mode (Text=t/CSV=c): ")
if mode not in ['t', 'c']:
    print("Invalid mode, aborting...")
    exit()
if mode == None:
    mode = 'c'
    
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model = model.to(device)
print("Model loaded")

def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)

class MessageDataset(Dataset):
    def __init__(self):
        super().__init__()
        if mode == "c":
            print("Running in CSV mode")
            # dataset_path = os.path.join('data', 'shortjokes.csv')
            dataset_path = os.path.join("data", 'trainingData.csv')

            self.data = []
            self.end_of_text_token = "<|endoftext|>"
            
            with open(dataset_path, encoding='utf-8') as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                
                x = 0
                for row in csv_reader:
                    message = f"MESSAGE:{row[1]}{self.end_of_text_token}"
                    self.data.append(message)
        elif mode == "t":
            print("Running in plain text mode")
            dataset_path = os.path.join('data', 'trainingData.txt')
            self.data = []
            self.end_of_text_token = "<|endoftext|>"
            with open(dataset_path, encoding='utf-8') as file:
                data = file.read()
                for message in data:
                    message = f"MESSAGE:{message}{self.end_of_text_token}"
                    self.data.append(message)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

dataset = MessageDataset()
dataLoader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
print("Created Dataset and DataLoader")

train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

trainLoader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
valLoader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

best_val_loss = float('inf')
early_stopping_counter = 0
EARLY_STOPPING_PATIENCE = 5

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=MAX_LR, steps_per_epoch=len(trainLoader), epochs=EPOCHS)

models_folder = "trained_models"
if not os.path.exists(models_folder):
    os.mkdir(models_folder)
print("Starting training")


for epoch in range(EPOCHS):
    model.train()
    print(f"EPOCH {epoch} started" + '=' * 30)
    
    progress_bar = tqdm.tqdm(enumerate(trainLoader), total=len(trainLoader))
    for idx,message in progress_bar:
        tens = torch.tensor(tokenizer.encode(message[0])).unsqueeze(0).to(device)
        if tens.size()[1] > MAX_SEQ_LEN:
            continue
        
        outputs = model(tens, labels=tens)
        loss, logits = outputs[:2]                        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        model.zero_grad()

        scheduler.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for idx, message in enumerate(valLoader):
            tens = torch.tensor(tokenizer.encode(message[0])).unsqueeze(0).to(device)
            if tens.size()[1] > MAX_SEQ_LEN:
                continue
            outputs = model(tens, labels=tens)
            loss, logits = outputs[:2]
            val_loss += loss.item()

    avg_val_loss = val_loss / len(valLoader)
    # avg_val_loss = val_loss
    print(f'Validation Loss: {avg_val_loss}')
    print(generate(prompt="Hello there, how are you doing?", model=model))

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model.save_pretrained(os.path.join(models_folder, UNIQUE_MODEL_NAME, 'best_model'))
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
            # break
            pass

    # Save model every 3 epochs
    if epoch % 3 == 0:
        model.save_pretrained(os.path.join(models_folder, UNIQUE_MODEL_NAME, f'epoch_{epoch}'))


model.save_pretrained(os.path.join(models_folder, UNIQUE_MODEL_NAME, 'final_model'))
# model.load_state_dict(torch.load(os.path.join(models_folder, 'best_model')))
print("Training finished")
print("Exiting program...")