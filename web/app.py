from flask import jsonify, render_template, request, redirect, url_for, send_file
from __init__ import app, db, login_manager
from models import User, Chat, Message
from flask_login import login_user, login_required, current_user, logout_user
import json, numpy as np, time, os, csv
from generate import generate, generate_with_history, generate_jokes

def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob)  # normalize to sum to 1
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)

@login_manager.user_loader
def login_manager(user_id):
    return User.query.get(user_id)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = ""
    if request.method == "POST":
        name = request.form.get('name')
        remember = request.form.get('remember')
        password = request.form.get('password')
        user = User.query.filter_by(name=name).first()
        if user is not None:
            if user.check_password(password):
                login_user(user, remember=remember)
                return redirect(url_for('chats'))
            else:
                error = 'Invalid password'
        else:
            error = 'User does not exist'
            
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = ""
    if request.method == "POST":
        print("Post")
        name = request.form.get('name')
        password = request.form.get('password')
        exists = User.query.filter_by(name=name).first()
        if exists:
            error = 'User already exists'
        else:
            user = User(name=name)
            user.hash_password(password)
            db.session.add(user)
            db.session.commit()
            return redirect(url_for('login'))
    return render_template('register.html', error=error)

@app.route('/chats')
@login_required
def chats():
    return render_template('chats.html', chats=Chat.query.filter_by(user_id=current_user.id).all())

@app.route('/chat/<int:id>')
@login_required
def chat(id):
    return render_template('chat.html', chat_id=int(id))

@app.route('/chat/joker/<int:id>')
def jokechat(id):
    return render_template('jokes_chat.html', chat_id=int(id))


@app.route('/chat/create', methods=['GET', 'POST'])
@login_required
def create_chat():
    if request.method == 'POST':
        newChat = Chat(user_id=current_user.id, model=request.form.get('model'), name=request.form.get('name'), is_joke=False)
        db.session.add(newChat)
        db.session.commit()
        return redirect(url_for('chat', id=newChat.id))
    
    chat_models = os.listdir('models')
    for chat in chat_models:
        if chat == 'Jokes':
            chat_models.remove(chat)
    
    return render_template('create_chat.html', chat_models=chat_models)
    # newChat = Chat(user_id=current_user.id)
    # db.session.add(newChat)
    # db.session.commit()
    # return redirect(url_for('chat', id=newChat.id))

@app.route('/chat/create/joker', methods=['GET', 'POST'])
def create_joker():
    if request.method == 'POST':
        newChat = Chat(user_id=current_user.id, model=request.form.get('model'), name=request.form.get('name'), is_joke=True)
        db.session.add(newChat)
        db.session.commit()
        return redirect(url_for('jokechat', id=newChat.id))
    
    joke_chats = os.listdir(os.path.join('models', 'Jokes'))

    return render_template('create_joker.html', chat_models=joke_chats)


@app.route('/chat/joker/<int:id>/create', methods=['POST'])
def post_joke(id):
    chat = Chat.query.get(id)
    count = int(request.form.get('count'))
    jokes = generate_jokes(count, chat.model)
    for joke in jokes:
        newMessage = Message(chat_id=chat.id, message=joke, is_from='bot')
        db.session.add(newMessage)
    db.session.commit()
    return get_chat(id)
    

@app.route('/chat/<int:id>/get')
def get_chat(id):
    output = ""
    for message in Chat.query.get(id).show_messages():
        output += ("<strong>User: " if message.is_from == 'user' else "") + message.message + ("</strong>"if message.is_from == 'user' else "") + "<br> <br>"
    return output

@app.route('/chat/<int:id>/send/joke', methods=['POST'])
def post_joke_message(id):
    chat = Chat.query.get(id)
    count = request.form.get('count')
    jokes = generate_jokes(count, chat.model)
    for joke in jokes:
        newMessage = Message(chat_id=chat.id, message=joke, is_from='bot')
        db.session.add(newMessage)
    db.session.commit()
    return get_chat(id)

@app.route('/chat/<int:id>/send', methods=['POST'])
def post_message(id):
    chat = Chat.query.get(id)
    message = request.form.get('message')
    print(message)
    full_history = request.form.get('full_history')
    newMessage = Message(chat_id=chat.id, message=message, is_from='user')
    history = Chat.query.get(id).show_messages()
    if full_history:
        response = generate_with_history(prompt=message, history=history)
    else:
        response = generate(prompt=message, model = chat.model)
    newResponse = Message(chat_id=chat.id, message=response, is_from='bot')
    db.session.add(newMessage)
    db.session.add(newResponse)   
    db.session.commit()
    return get_chat(id)

@app.route('/chat/delete/<int:id>')
@login_required
def chat_delete(id):
    chat = Chat.query.filter_by(id=id).first()
    if chat:
        if current_user.id == chat.user_id:
            Message.query.filter_by(chat_id=id).delete()  # Delete associated messages
            db.session.delete(chat)
            db.session.commit()
    return redirect(url_for('chats'))

@app.route('/datasets')
def datasets():
    datasets = os.listdir('datasets')
    datasets_dict = []
    for id, dataset in enumerate(datasets):
        datasets_dict.append({
            'name': dataset,
            'id': id,
            'size': os.path.getsize(os.path.join('datasets', dataset))
        })

    return render_template('datasets.html', datasets=datasets_dict)

# @app.route('/dataset/<int:id>')
# def get_dataset(id):
#     path = os.path.join('datasets', os.listdir('datasets')[id])
#     print(path)
#     return send_from_directory(path, 'dataset.csv')

@app.route('/dataset/download/<int:id>')
def download_dataset(id):
    path = os.path.join(os.getcwd(), 'datasets', os.listdir('datasets')[id])
    print(path)
    
    return send_file(path, as_attachment=True, download_name=os.listdir('datasets')[id])

@app.route('/dataset/humanized/<int:id>')
def get_humanized_dataset(id):
    with open(os.path.join('datasets', os.listdir('datasets')[id]), 'r', encoding='utf-8') as file:
        data_reader = csv.reader(file, delimiter=',')
        data = []
        for row in data_reader:
            data.append(row)

        output = ""
        for i in range(1, 101):
            output += f"Joke {i}: {data[i][1]}<br>"

    return output


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


if __name__ == '__main__':
    # app.run(host='0.0.0.0', ssl_context='adhoc', debug=True)
    app.run(port=5001, debug=True)
    
    # 195.240.271.189