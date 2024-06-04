import os, torch, numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob)  # normalize to sum to 1
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)


def generate(prompt, model = None):
    if model is None:
        model = GPT2LMHeadModel.from_pretrained('./models/gpt2_medium_17')
    else:
        model = GPT2LMHeadModel.from_pretrained(os.path.join('models', model))
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    model.eval()
    
    joke_num = 0
    with torch.no_grad():
        
        joke_finished = False
        # input = "USER: " + prompt + "<|endoftext|> \n BOT: "
        cur_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
        new_text = ""
        for i in range(100):
            outputs = model(cur_ids, labels=cur_ids)
            loss, logits = outputs[:2]
            softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(from only one in this case) batch and the last predicted embedding
            if i < 3:
                n = 20
            else:
                n = 3
            next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=n) #Randomly(from the topN probability distribution) select the next word
            print(tokenizer.decode([next_token_id]))
            new_text += tokenizer.decode([next_token_id])
            cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1) # Add the last word to the running sequence

            if next_token_id in tokenizer.encode('<|endoftext|>'):
                joke_finished = True
                break

        
        if joke_finished:
            
            joke_num = joke_num + 1
            
            output_list = list(cur_ids.squeeze().to('cpu').numpy())
            output_text = tokenizer.decode(output_list)

            print(output_text.split("<|endoftext|>")[0])
            return new_text
        else:
            return new_text
        
def generate_with_history(prompt, history):
    model = GPT2LMHeadModel.from_pretrained('./models/gpt2_medium_118')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    model.eval()

    def choose_from_top(probs, n=5):
        ind = np.argpartition(probs, -n)[-n:]
        top_prob = probs[ind]
        top_prob = top_prob / np.sum(top_prob)  # normalize to sum to 1
        choice = np.random.choice(n, 1, p = top_prob)
        token_id = ind[choice][0]
        return int(token_id)

    num = 0
    hist = ""
    with torch.no_grad():
            num = 0
            with torch.no_grad():
                message_finished = False
                for message in history:
                    user = "User: " if message.is_from == "user" else "Bot: "
                    hist += user + message.message + "<|endoftext|> \n"

                user_input = "User: " + prompt + "<|endoftext|> \n Bot: "
                hist += user_input

                cur_ids = torch.tensor(tokenizer.encode(hist)).unsqueeze(0).to(device)

                for i in range(100):
                    print(i)
                    outputs = model(cur_ids, labels=cur_ids)
                    loss, logits = outputs[:2]
                    softmax_logits = torch.softmax(logits[0,-1], dim=0)
                    if i < 3:
                        n = 20
                    else:
                        n = 3
                    next_token_id = choose_from_top(softmax_logits.to(device).numpy(), n=n)
                    cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1)
                    print(tokenizer.decode(cur_ids.squeeze().to(device).numpy()))
                    if next_token_id in tokenizer.encode('<|endoftext|>'):
                        message_finished = True
                        break

                if message_finished:
                    
                    num = num + 1
                    
                    output_list = list(cur_ids.squeeze().to(device).numpy())
                    output_text = tokenizer.decode(output_list)
                    output_text = output_text.split("<|endoftext|>")[1]
                    # output_text = output_text.replace("Bot: ", "")
                    print("---------------------------------")
                    print(output_text)

                    return output_text.split("<|endoftext|>")[0]

def generate_jokes(count, model):
    model = GPT2LMHeadModel.from_pretrained(os.path.join('models', 'Jokes', model)).to(device)
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

# # import os, torch, numpy as np, tqdm
# # from transformers import GPT2Tokenizer, GPT2LMHeadModel

# # for i in range(0, 24):
# #     print(i)
# #     # model = GPT2LMHeadModel.from_pretrained('trainedModel')
# #     # print(os.listdir('./models'))
# #     model = GPT2LMHeadModel.from_pretrained('./models/gpt2_medium_'+str(i))
# #     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# #     model.to(device)

# #     MODEL_EPOCH = 4

# #     models_folder = "trained_models"

# #     output_file_path = f'generated_{MODEL_EPOCH}.txt'

# #     model.eval()
# #     if os.path.exists(output_file_path):
# #         os.remove(output_file_path)
        
# #     def choose_from_top(probs, n=5):
# #         ind = np.argpartition(probs, -n)[-n:]
# #         top_prob = probs[ind]
# #         top_prob = top_prob / np.sum(top_prob)  # normalize to sum to 1
# #         choice = np.random.choice(n, 1, p = top_prob)
# #         token_id = ind[choice][0]
# #         return int(token_id)

# #     num = 0
# #     with torch.no_grad():
# #         history = "Bot: How are you doing today?"
# #         # print(history)
# #         # while True:
# #         message_finished = False
# #         # user_input = input("Enter your message: ")
# #         user_input = "Oh hello there, how are you? \n"

# #         # user_input = "Oh hello there, how are you? \n"
# #         history += "User: " + user_input + " <|endoftext|>" + "\n" + "Bot: "
        
# #         # print(tokenizer.decode(cur_ids.squeeze().to(device).numpy()))
# #         new_message = ""
# #         for i in range(1000):
# #             cur_ids = torch.tensor(tokenizer.encode(history)).unsqueeze(0).to(device)
# #             # print("=========================================")
# #             # print(history)
# #             # print("=========================================")
# #             outputs = model(cur_ids, labels=cur_ids)
# #             loss, logits = outputs[:2]
# #             softmax_logits = torch.softmax(logits[0,-1], dim=0)
# #             if i < 3:
# #                 n = 20
# #             else:
# #                 n = 3
# #             next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=n)
# #             next_text = tokenizer.decode([next_token_id])
# #             # print(next_text)
# #             # if next_text == "<|endoftext|>" and i < 10:
# #             #     continue
# #             history += next_text
# #             new_message += next_text
# #             # print(tokenizer.decode(next_token_id))
# #             # cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1)
            
        
# #             if next_token_id in tokenizer.encode('<|endoftext|>'):
# #                 message_finished = True
# #                 history += "\n"
# #                 print(new_message)
# #                 # print(new_message.split("<|endoftext|>")[0])
# #                 with open(output_file_path, 'a') as f:
# #                     f.write(str(i) + new_message.split("<|endoftext|>")[0] + "\n")
# #                 break

# import os, torch, numpy as np, tqdm
# from transformers import GPT2Tokenizer, GPT2LMHeadModel

# model = GPT2LMHeadModel.from_pretrained('./models/Jokes')
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model.to(device)

# models_folder = "trained_models"

# # model.load_state_dict(torch.load(model_path))

# output_file_path = f'output.txt'

# model.eval()
# if os.path.exists(output_file_path):
#     os.remove(output_file_path)
    
# def choose_from_top(probs, n=5):
#     ind = np.argpartition(probs, -n)[-n:]
#     top_prob = probs[ind]
#     top_prob = top_prob / np.sum(top_prob)  # normalize to sum to 1
#     choice = np.random.choice(n, 1, p = top_prob)
#     token_id = ind[choice][0]
#     return int(token_id)

# num = 0
# with torch.no_grad():
#         # progress = tqdm.tqdm(range(100))
#         # for idx in progress:
        
#             num = 0
#             with torch.no_grad():
#                 # history = ""
#                 while True:  # Keep the conversation going until the user decides to stop
#                     message_finished = False
#                     user_input = input("You: ")  # Get the user's input
#                     # history += "User: " + user_input + "<|endoftext|>" + "\n" + "Bot: "

#                     # If the user types 'quit', end the conversation
#                     if user_input.lower() == 'quit':
#                         break

#                     cur_ids = torch.tensor(tokenizer.encode(user_input)).unsqueeze(0).to(device)

#                     for i in range(1000):
#                         outputs = model(cur_ids, labels=cur_ids)
#                         loss, logits = outputs[:2]
#                         softmax_logits = torch.softmax(logits[0,-1], dim=0)
#                         if i < 3:
#                             n = 20
#                         else:
#                             n = 3
#                         next_token_id = choose_from_top(softmax_logits.to(device).numpy(), n=n)
#                         cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1)

#                         if next_token_id in tokenizer.encode('<|endoftext|>'):
#                             message_finished = True
#                             break

#                     if message_finished:
                        
#                         num = num + 1
                        
#                         output_list = list(cur_ids.squeeze().to(device).numpy())
#                         output_text = tokenizer.decode(output_list)
#                         # history += output_text + "\n"
#                         # print(history)

#                         with open(output_file_path, 'a') as f:
#                             f.write(f"{output_text} \n\n")



#             # cur_ids = torch.tensor(tokenizer.encode("Message:")).unsqueeze(0).to(device)

#             # for i in range(1000):
#             #     outputs = model(cur_ids, labels=cur_ids)
#             #     loss, logits = outputs[:2]
#             #     softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(from only one in this case) batch and the last predicted embedding
#             #     if i < 3:
#             #         n = 20
#             #     else:
#             #         n = 3
#             #     next_token_id = choose_from_top(softmax_logits.to(device).numpy(), n=n) #Randomly(from the topN probability distribution) select the next word
#             #     cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1) # Add the last word to the running sequence

#             #     if next_token_id in tokenizer.encode('<|endoftext|>'):
#             #         message_finished = True
#             #         break

            
            