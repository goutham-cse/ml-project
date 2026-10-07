import os
import sqlite3
import joblib
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file
from werkzeug.security import generate_password_hash, check_password_hash

# Import custom ML classes so joblib unpickling can resolve class attributes
from train_model import (
    RidgeRegressor,
    SoftmaxClassifier,
    PureStandardScaler,
    PureKNN,
    GradientBoostingRegressorPure
)

from github_linkedin_analyzer import analyze_github_profile, analyze_linkedin_profile
from feedback_engine import generate_feedback

app = Flask(__name__)
app.secret_key = 'antigravity_placement_predictor_secret_key'

DB_PATH = 'users.db'

# ==========================================
# Database Initialization
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Load ML artifacts
def load_ml_model():
    if os.path.exists('placement_model.joblib'):
        return joblib.load('placement_model.joblib')
    else:
        from train_model import train_and_evaluate_all
        return train_and_evaluate_all()

artifacts = load_ml_model()

# ==========================================
# Routes & Views
# ==========================================

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip()
        password = request.form['password'].strip()

        if not username or not email or not password:
            flash('Please fill in all fields.', 'error')
            return redirect(url_for('register'))

        hashed_pw = generate_password_hash(password)

        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
                           (username, email, hashed_pw))
            conn.commit()
            conn.close()
            flash('Account created successfully! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or Email already exists.', 'error')
            return redirect(url_for('register'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username_or_email = request.form['username'].strip()
        password = request.form['password'].strip()

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? OR email = ?', (username_or_email, username_or_email))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):
            session['user_id'] = user[0]
            session['username'] = user[1]
            flash(f'Welcome back, {user[1]}!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid username/email or password.', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))


@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', username=session.get('username'))


@app.route('/api/fetch_github', methods=['POST'])
def fetch_github_api():
    data = request.get_json() or {}
    github_handle = data.get('github_handle', '')
    gh_metrics = analyze_github_profile(github_handle)
    return jsonify(gh_metrics)


@app.route('/predict', methods=['POST'])
def predict():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    try:
        name = request.form.get('name', session.get('username'))
        dept = request.form.get('dept', 'Computer Science')
        cgpa = float(request.form.get('cgpa', 7.5))
        skills_str = request.form.get('skills', 'Python, Data Structures, Web Development')
        
        leetcode_rating = int(request.form.get('leetcode_rating', 1200))
        leetcode_solved = int(request.form.get('leetcode_solved', 150))
        
        github_handle = request.form.get('github_handle', '')
        gh_metrics = analyze_github_profile(github_handle)
        github_repos = int(request.form.get('github_repos', gh_metrics['repos']))
        github_commits_year = int(request.form.get('github_commits_year', gh_metrics['commits_year']))
        github_stars = int(request.form.get('github_stars', gh_metrics['stars']))
        github_score = float(gh_metrics['github_score'])

        linkedin_connections = int(request.form.get('linkedin_connections', 200))
        linkedin_certs = int(request.form.get('linkedin_certs', 2))
        linkedin_posts_freq = int(request.form.get('linkedin_posts_freq', 5))
        li_metrics = analyze_linkedin_profile(linkedin_connections, linkedin_certs, linkedin_posts_freq)
        linkedin_score = float(li_metrics['linkedin_score'])

        projects_count = int(request.form.get('projects_count', 2))
        project_complexity = request.form.get('project_complexity', 'Intermediate')
        
        complexity_map = {'Basic': 25, 'Intermediate': 55, 'Advanced': 80, 'Production-Grade': 100}
        comp_val = complexity_map.get(project_complexity, 55)
        project_score = round(min(100.0, max(10.0, (projects_count / 6 * 40) + (comp_val * 0.6))), 1)

        skills_list = [s.strip() for s in skills_str.split(',') if s.strip()]
        skills_count = len(skills_list)
        skill_score = round(min(100.0, max(15.0, skills_count * 8.5 + 20.0)), 1)

        coding_score = round(min(100.0, max(0.0, (leetcode_rating - 800) / 1400 * 60 + (leetcode_solved / 500 * 40))), 1)
        cgpa_score = round((cgpa / 10.0) * 100, 1)
        dept_boost = 5.0 if dept in ['Computer Science', 'Information Technology'] else (2.0 if dept == 'Electronics & Comm' else 0.0)
        comp_enc = artifacts['comp_map'].get(project_complexity, 1)

        feat_vector = np.array([[
            cgpa, float(skills_count), skill_score, float(leetcode_rating), float(leetcode_solved),
            float(github_repos), float(github_commits_year), float(github_stars), github_score,
            float(linkedin_connections), float(linkedin_certs), float(linkedin_posts_freq), linkedin_score,
            float(projects_count), float(comp_enc), project_score, dept_boost,
            coding_score, cgpa_score
        ]])

        scaled_feat = artifacts['scaler'].transform(feat_vector)
        pred_score = float(artifacts['ridge_reg'].predict(scaled_feat)[0])
        pred_score = round(min(99.5, max(5.0, pred_score)), 1)

        if pred_score < 52.0:
            tier_id = 0
            tier_label = "Not Ready (< 52%)"
            tier_badge_class = "badge-danger"
            tier_desc = "Significant gaps in coding or fundamentals. Immediate targeted practice needed."
        elif pred_score < 68.0:
            tier_id = 1
            tier_label = "Service Tier Ready (52% - 68%)"
            tier_badge_class = "badge-warning"
            tier_desc = "Eligible for Service/IT Consultancy hiring drives. Needs DSA push for Product Tier."
        elif pred_score < 83.0:
            tier_id = 2
            tier_label = "Product Tier Ready (68% - 83%)"
            tier_badge_class = "badge-info"
            tier_desc = "Strong foundation for Core Product Companies (6-15 LPA)."
        else:
            tier_id = 3
            tier_label = "Dream Tier / Tier-1 Ready (≥ 83%)"
            tier_badge_class = "badge-success"
            tier_desc = "Top candidate for Tier-1 Product & High Package Dream Offers (15-40+ LPA)."

        feedback = generate_feedback(
            cgpa, leetcode_rating, leetcode_solved, github_score, github_repos,
            linkedin_score, linkedin_certs, projects_count, project_complexity,
            skills_count, dept, pred_score
        )

        result_data = {
            'name': name,
            'dept': dept,
            'cgpa': cgpa,
            'skills_list': skills_list,
            'skills_count': skills_count,
            'leetcode_rating': leetcode_rating,
            'leetcode_solved': leetcode_solved,
            'github_handle': github_handle,
            'github_repos': github_repos,
            'github_commits_year': github_commits_year,
            'github_stars': github_stars,
            'github_score': github_score,
            'github_languages': gh_metrics.get('languages', []),
            'linkedin_connections': linkedin_connections,
            'linkedin_certs': linkedin_certs,
            'linkedin_score': linkedin_score,
            'projects_count': projects_count,
            'project_complexity': project_complexity,
            'project_score': project_score,
            'coding_score': coding_score,
            'cgpa_score': cgpa_score,
            'readiness_score': pred_score,
            'tier_id': tier_id,
            'tier_label': tier_label,
            'tier_badge_class': tier_badge_class,
            'tier_desc': tier_desc,
            'strengths': feedback['strengths'],
            'weaknesses': feedback['weaknesses'],
            'recommendations': feedback['recommendations'],
            'summary': feedback['summary']
        }

        return render_template('result.html', r=result_data, username=session.get('username'))

    except Exception as e:
        flash(f"Error making prediction: {str(e)}", "error")
        return redirect(url_for('index'))


@app.route('/analytics')
def analytics():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    return render_template('analytics.html', 
                           cls_summary=artifacts['cls_summary'],
                           reg_summary=artifacts['reg_summary'],
                           best_cls_name=artifacts['best_cls_name'],
                           best_cls_acc=round(artifacts['best_cls_acc']*100, 2),
                           feature_importances=artifacts['feature_importances'],
                           username=session.get('username'))


@app.route('/export/<filename>')
def export_file(filename):
    if 'user_id' not in session:
        return redirect(url_for('login'))

    allowed = ['placement_dataset.csv', 'placement_dataset_info.xlsx', 'ml_model_report.md', 'placement_ml_report.pdf']
    if filename in allowed and os.path.exists(filename):
        return send_file(filename, as_attachment=True)
    else:
        flash('Requested file not found.', 'error')
        return redirect(url_for('analytics'))


if __name__ == '__main__':
    print("Launching Placement Readiness Predictor Flask Server on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
