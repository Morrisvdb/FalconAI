from flask_login import UserMixin
from flask_bcrypt import Bcrypt
from __init__ import db, app

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    password = db.Column(db.String(120), nullable=False)
    
    
    def hash_password(self, password):
        self.password = Bcrypt().generate_password_hash(password).decode('utf-8')
        
    def check_password(self, password):
        return Bcrypt().check_password_hash(self.password, password)
    
    def __repr__(self):
        return f'<User {self.id}>'
    

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.String(120), nullable=False)
    is_from = db.Column(db.String(32)) # user/bot 
    chat_id = db.Column(db.Integer, db.ForeignKey('chat.id'), nullable=False)
    
    def __repr__(self):
        return f'<Message {self.id}>'
    
class Chat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    messages = db.relationship('Message', backref='chat', lazy=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(120), nullable=True)
    model = db.Column(db.String(32), nullable=False)
    is_joke = db.Column(db.Boolean, default=False)
    
    def show_messages(self):
        return Message.query.filter_by(chat_id=self.id).all()
        
    def __repr__(self):
        return f'<Chat {self.id}>'
    
    
with app.app_context():
    db.create_all()