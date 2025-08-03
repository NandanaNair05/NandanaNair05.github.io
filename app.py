from flask import (
    Flask, render_template, request, redirect,
    url_for, flash
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, login_required,
    logout_user, current_user
)
from datetime import datetime
import os
import cv2

app = Flask(__name__)
app.secret_key = 'change_this_secret_key'

# ── Database ──────────────────────────────────────────────────────
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

# ── Models ────────────────────────────────────────────────────────
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)  # plaintext for demo

class Scan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    mask_path = db.Column(db.String(300))
    contrast_path = db.Column(db.String(300))
    result_path = db.Column(db.String(300))
    edge_path = db.Column(db.String(300))
    user = db.relationship('User', backref=db.backref('scans', lazy=True))

# ── Login manager ────────────────────────────────────────────────
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ── Ensure DB exists on every request ─────────────────────────────
@app.before_request
def create_tables():
    db.create_all()

# ── File‑upload path ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ── Auth routes ──────────────────────────────────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        db.session.add(User(username=username, password=password))
        db.session.commit()
        flash('User created! Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.password == password:
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/tutorials')
@login_required
def tutorials():
    return render_template('tutorials.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully')
    return redirect(url_for('login'))

# ── Main pages ───────────────────────────────────────────────────
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html', username=current_user.username)

@app.route('/about')
@login_required
def about():
    return render_template('about.html', username=current_user.username)

@app.route('/')
@login_required
def home():
    latest_scan = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.timestamp.desc()).first()

    def file_exists(filename):
        if not filename:
            return False
        return os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    if latest_scan and all([
        file_exists(latest_scan.mask_path),
        file_exists(latest_scan.contrast_path),
        file_exists(latest_scan.result_path),
        file_exists(latest_scan.edge_path)
    ]):
        return render_template('index.html', username=current_user.username, scan=latest_scan)

    return render_template('index.html', username=current_user.username, scan=None)

@app.route('/scans')
@login_required
def scans():
    user_scans = Scan.query.filter_by(user_id=current_user.id).order_by(Scan.timestamp.desc()).all()
    return render_template('scans.html', scans=user_scans, username=current_user.username)

@app.route('/clear_results', methods=['POST'])
@login_required
def clear_results():
    scans = Scan.query.filter_by(user_id=current_user.id).all()

    for scan in scans:
        for filename in [scan.result_path, scan.edge_path]:
            if filename:
                full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if os.path.exists(full_path):
                    os.remove(full_path)
        scan.result_path = None
        scan.edge_path = None

    db.session.commit()
    flash("Processed DSA results cleared! Original mask and contrast images are kept.")
    return redirect(url_for('home'))

# ── DSA upload / processing ──────────────────────────────────────
@app.route('/upload', methods=['POST'])
@login_required
def upload_files():
    if 'mask' not in request.files or 'contrast' not in request.files:
        flash('Please upload both mask and contrast images')
        return redirect(url_for('home'))

    mask_file = request.files['mask']
    contrast_file = request.files['contrast']

    mask_filename = 'mask.jpeg'
    contrast_filename = 'contrast.jpeg'
    result_filename = 'subtracted_result.jpeg'
    edge_filename = 'vessels_edges.jpeg'

    mask_path = os.path.join(app.config['UPLOAD_FOLDER'], mask_filename)
    contrast_path = os.path.join(app.config['UPLOAD_FOLDER'], contrast_filename)

    mask_file.save(mask_path)
    contrast_file.save(contrast_path)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    contrast = cv2.imread(contrast_path, cv2.IMREAD_GRAYSCALE)
    contrast = cv2.resize(contrast, (mask.shape[1], mask.shape[0]))

    mask_blur = cv2.GaussianBlur(mask, (5, 5), 0)
    contrast_blur = cv2.GaussianBlur(contrast, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    mask_eq = clahe.apply(mask_blur)
    contrast_eq = clahe.apply(contrast_blur)

    subtracted = cv2.absdiff(contrast_eq, mask_eq)
    # Normalize the subtracted image to enhance vessel visibility
    subtracted = cv2.normalize(subtracted, None, 0, 255, cv2.NORM_MINMAX)

    edges = cv2.Canny(subtracted, 50, 150)

    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], result_filename), subtracted)
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], edge_filename), edges)

    new_scan = Scan(
        user_id=current_user.id,
        mask_path=mask_filename,
        contrast_path=contrast_filename,
        result_path=result_filename,
        edge_path=edge_filename
    )
    db.session.add(new_scan)
    db.session.commit()

    flash('DSA simulation completed!')
    return redirect(url_for('home'))

# ── Run ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True)
