from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_talisman import Talisman

app = Flask(__name__)
# talisman = Talisman(app)

app.config['SECRET_KEY'] = 'kjhsdlfwuof_gwidgfjadsgfodsgifugadskfjas'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite'


db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


# csp = {
# 	'default-src': [
# 		'\'self\'',
# 		'https://code.jquery.com',
# 		'https://cdn.jsdelivr.net',
#         'https://unpkg.com'
# 	]
# }
# # HTTP Strict Transport Security (HSTS) Header
# hsts = {
# 	'max-age': 31536000,
# 	'includeSubDomains': True
# }
# # Enforce HTTPS and other headers
# talisman.force_https = True
# talisman.force_file_save = True
# talisman.x_xss_protection = True
# talisman.session_cookie_secure = True
# talisman.session_cookie_samesite = 'Lax'
# talisman.frame_options_allow_from = 'https://www.google.com'

# # Add the headers to Talisman
# talisman.content_security_policy = csp
# talisman.strict_transport_security = hsts
