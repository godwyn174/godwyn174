from flask import Flask, render_template, request, redirect, url_for, session, flash, make_response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from itsdangerous import URLSafeTimedSerializer
import os
from datetime import timedelta

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secure session key
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)  # Session duration for "Remember Me"

# PostgreSQL configuration (update with your Supabase/local credentials)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:your_password@localhost/soil_health_db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Secure token serializer for "Remember Me"
serializer = URLSafeTimedSerializer(app.secret_key)

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'farmer' or 'admin'
    remember_token = db.Column(db.String(256), nullable=True)  # Token for "Remember Me"

# Create database tables
with app.app_context():
    db.create_all()

@app.route('/')
def index():
    # Check for remember token in cookies
    token = request.cookies.get('remember_token')
    if token and 'user_id' not in session:
        try:
            user_id = serializer.loads(token, max_age=604800)  # 7 days
            user = User.query.get(user_id)
            if user and user.remember_token == token:
                session['user_id'] = user.id
                session['username'] = user.username
                session['role'] = user.role
                session.permanent = True
                return redirect(url_for('dashboard'))
        except:
            pass
    return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        remember = 'remember' in request.form  # Check if "Remember Me" is selected
        
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['username'] = user.username
            session['role'] = user.role
            session.permanent = remember  # Set session duration
            
            # Generate and store remember token if checked
            response = make_response(redirect(url_for('dashboard')))
            if remember:
                token = serializer.dumps(user.id)
                user.remember_token = token
                db.session.commit()
                response.set_cookie('remember_token', token, max_age=604800, httponly=True, secure=True)
            else:
                user.remember_token = None
                db.session.commit()
                response.delete_cookie('remember_token')
            
            flash('Login successful!', 'success')
            return response
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        role = request.form['role']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'danger')
            return render_template('signup.html')
        
        # Hash password with SHA-256
        hashed_password = generate_password_hash(password, method='sha256')
        new_user = User(username=username, password=hashed_password, role=role)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        flash('Please log in to access the dashboard', 'danger')
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['username'], role=session['role'])

@app.route('/logout')
def logout():
    user = User.query.get(session.get('user_id'))
    if user:
        user.remember_token = None
        db.session.commit()
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('role', None)
    response = make_response(redirect(url_for('login')))
    response.delete_cookie('remember_token')
    flash('Logged out successfully', 'success')
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)