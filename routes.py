import os
import json
from datetime import datetime, timezone
import time
from functools import wraps
from urllib.parse import urlparse

from flask import render_template, request, redirect, url_for, flash, jsonify, abort, session, g, send_from_directory
from werkzeug.utils import secure_filename

from app import app, limiter, celery, supabase
from data_access import (
    get_user_by_id, get_user_by_email, create_user, update_user,
    create_website, get_websites_by_user, get_website_by_id,
    create_report, get_reports_by_user, get_report_by_id, get_report_by_filename, update_report, mark_report_as_paid,
    create_analysis_job, get_analysis_job_by_filename, get_analysis_jobs_by_user, update_analysis_job, update_job_progress, complete_analysis_job,
    create_task, get_tasks_by_report, update_task, complete_task
)
from storage import get_report_from_storage, save_report_to_storage
from tasks import run_seo_analysis

# Basic authentication function using Supabase
def get_auth_user():
    """Get the authenticated user from the session"""
    auth_token = session.get('auth_token')
    if not auth_token:
        return None
    
    try:
        # Verify the token with Supabase
        user = supabase.auth.get_user(auth_token)
        return user.user if hasattr(user, 'user') else None
    except Exception as e:
        app.logger.error(f"Authentication error: {e}")
        return None

# Authentication decorators
def auth_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_auth_user()
        if not user:
            flash('Please log in to access this page', 'error')
            return redirect(url_for('login'))
        
        g.user = user
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        user = get_auth_user()
        if not user:
            flash('Please log in to access this page', 'error')
            return redirect(url_for('login'))
        
        # Check if user is admin
        db_user = get_user_by_email(user.email)
        if not db_user or not db_user.get('is_admin'):
            flash('Admin privileges required', 'error')
            return redirect(url_for('dashboard'))
        
        g.user = user
        return f(*args, **kwargs)
    return decorated

# User helper functions
def get_user_email():
    """Get the email of the authenticated user"""
    user = getattr(g, 'user', None)
    return user.email if user else None

def get_user_username():
    """Get the username of the authenticated user"""
    user = getattr(g, 'user', None)
    if not user:
        return None
    
    db_user = get_user_by_email(user.email)
    if db_user and db_user.get('username'):
        return db_user.get('username')
    
    # If no username is set, use the first part of the email
    return user.email.split('@')[0]

# Main routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # This would normally redirect to Supabase Auth UI
        # For now, we'll just simulate a successful signup
        flash('Please sign up using Supabase Auth', 'info')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # This would normally redirect to Supabase Auth UI
        # For now, we'll just simulate a successful login
        flash('Please log in using Supabase Auth', 'info')
        return render_template('login.html')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('auth_token', None)
    flash('You have been logged out', 'success')
    return redirect(url_for('index'))

@app.route('/dashboard')
@auth_required
def dashboard():
    reports = get_reports_by_user(g.user.id)
    return render_template('dashboard.html', 
                          reports=reports,
                          username=get_user_username())

@app.route('/profile')
@auth_required
def profile():
    user = get_user_by_email(get_user_email())
    return render_template('profile.html', user=user)

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    return render_template('admin_dashboard.html')

@app.route('/my-reports')
@auth_required
def my_reports():
    jobs = get_analysis_jobs_by_user(g.user.id)
    return render_template('my_reports.html', reports=jobs)

@app.route('/new-analysis', methods=['GET', 'POST'])
@auth_required
def new_analysis():
    if request.method == 'POST':
        url = request.form.get('url')
        if not url:
            flash('Please enter a valid URL', 'error')
            return redirect(url_for('new_analysis'))
        
        # Validate URL format
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Get the number of pages to analyze
        pages = request.form.get('pages', 5, type=int)
        
        try:
            # Create a new analysis job
            job = create_analysis_job(g.user.id, url, pages)
            if not job:
                flash('Error creating analysis job', 'error')
                return redirect(url_for('new_analysis'))
            
            filename = job.get('filename')
            
            try:
                # Try to start the Celery task
                task = run_seo_analysis.delay(url, pages, g.user.id, filename)
                
                # Update the job with the task ID
                update_analysis_job(filename, task_id=task.id)
            except Exception as celery_error:
                # Log the Celery error but don't crash
                app.logger.error(f"Celery task could not be started: {celery_error}")
                flash("Analysis started but background processing may be delayed. Please check back later.", "warning")
                update_analysis_job(filename, status='pending', current_step='queued')
            
            # Redirect to results page
            return redirect(url_for('analysis_results', filename=filename))
        except Exception as e:
            flash(f'Error starting analysis: {str(e)}', 'error')
            return redirect(url_for('new_analysis'))
    
    return render_template('new_analysis.html')

@app.route('/results/<filename>')
@auth_required
def analysis_results(filename):
    # Verify that the job exists and belongs to the current user
    job = get_analysis_job_by_filename(filename)
    if not job:
        flash('Analysis not found', 'error')
        return redirect(url_for('my_reports'))
    
    if job.get('user_id') != g.user.id:
        flash('You do not have permission to view this analysis', 'error')
        return redirect(url_for('my_reports'))
    
    return render_template('results.html', filename=filename)

@app.route('/api/progress/<filename>')
@auth_required
def get_progress(filename):
    """Endpoint to check the progress of an analysis"""
    job = get_analysis_job_by_filename(filename)
    
    if not job:
        return jsonify({"status": "not_found", "message": "Analysis job not found"}), 404
    
    # Verify that the job belongs to the current user
    if job.get('user_id') != g.user.id:
        return jsonify({"status": "unauthorized", "message": "Not authorized to access this job"}), 403
    
    # Convert datetime strings to timestamps
    if job.get('start_time'):
        try:
            job['start_time'] = datetime.fromisoformat(job['start_time']).timestamp()
        except ValueError:
            job['start_time'] = None
    
    if job.get('end_time'):
        try:
            job['end_time'] = datetime.fromisoformat(job['end_time']).timestamp()
        except ValueError:
            job['end_time'] = None
    
    # Return the current progress information
    return jsonify(job)

@app.route('/api/results/<filename>')
@auth_required
def get_results(filename):
    # Verify that the job exists and belongs to the current user
    job = get_analysis_job_by_filename(filename)
    if not job:
        return jsonify({"status": "not_found", "error": "Results file not found"}), 404
    
    if job.get('user_id') != g.user.id:
        return jsonify({"status": "unauthorized", "error": "Not authorized to access this job"}), 403
    
    # Check job status
    if job.get('status') in ['pending', 'running']:
        return jsonify({"status": "pending", "message": "Analysis in progress"}), 202
    
    if job.get('status') == 'failed':
        return jsonify({"status": "failed", "error": job.get('error', "Unknown error")}), 500
    
    try:
        # Get the report URL from job data
        report_url = job.get('report_url')
        
        if report_url:
            # Use our improved URL content fetcher that has fallback mechanisms
            from app import get_url_content
            content = get_url_content(report_url)
            
            if content:
                try:
                    # Try to parse as JSON
                    return jsonify(json.loads(content))
                except json.JSONDecodeError:
                    # If it's not valid JSON (e.g., HTML content), return as raw text
                    return content, 200, {'Content-Type': 'text/html'}
        
        # If no URL in job data, fall back to the old method
        data = get_report_from_storage(filename)
        if data:
            return jsonify(data)
        
        # If we can't get the file from storage, return an error
        return jsonify({"status": "not_found", "error": "Results file not found"}), 404
    except Exception as e:
        app.logger.error(f"Error fetching results for {filename}: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "error": f"Unexpected error: {str(e)}"}), 500

@app.route('/view-report/<report_id>')
@auth_required
def view_report(report_id):
    report = get_report_by_id(report_id)
    if not report:
        flash('Report not found', 'error')
        return redirect(url_for('dashboard'))
    
    if report.get('user_id') != g.user.id and not get_user_by_email(get_user_email()).get('is_admin', False):
        flash('You do not have permission to view this report', 'error')
        return redirect(url_for('dashboard'))
    
    # Check if the report is paid or the user is an admin
    if not report.get('is_paid') and not get_user_by_email(get_user_email()).get('is_admin', False):
        return redirect(url_for('analysis_results', filename=report.get('filename')))
    
    try:
        # First try to get the content using the report URL if available
        if report.get('report_url'):
            from app import get_url_content
            content = get_url_content(report.get('report_url'))
            if content:
                try:
                    # Try to parse as JSON
                    data = json.loads(content)
                    return render_template('view_report.html', report=report, data=data)
                except json.JSONDecodeError:
                    # If it's HTML content, we can render it directly
                    if '<html' in content:
                        return content
        
        # Fall back to the original method
        data = get_report_from_storage(report.get('filename'))
        return render_template('view_report.html', report=report, data=data)
    except Exception as e:
        app.logger.error(f"Error viewing report {report_id}: {str(e)}", exc_info=True)
        flash(f"Error loading report: {str(e)}", 'error')
        return redirect(url_for('dashboard'))

@app.route('/download-report/<report_id>')
@auth_required
def download_report(report_id):
    report = get_report_by_id(report_id)
    if not report:
        flash('Report not found', 'error')
        return redirect(url_for('dashboard'))
    
    if report.get('user_id') != g.user.id and not get_user_by_email(get_user_email()).get('is_admin', False):
        flash('You do not have permission to download this report', 'error')
        return redirect(url_for('dashboard'))
    
    # Check if the report is paid or the user is an admin
    if not report.get('is_paid') and not get_user_by_email(get_user_email()).get('is_admin', False):
        flash('You need to purchase this report to download it', 'error')
        return redirect(url_for('analysis_results', filename=report.get('filename')))
    
    try:
        filename = report.get('filename')
        content = None
        
        # Try to get content using improved method first
        if report.get('report_url'):
            from app import get_url_content
            content = get_url_content(report.get('report_url'))
        
        # Fall back to the original method if needed
        if not content:
            data = get_report_from_storage(filename)
            if isinstance(data, dict):
                content = json.dumps(data)
            else:
                content = data
        
        # If we still don't have content, show an error
        if not content:
            flash('Report data not found', 'error')
            return redirect(url_for('dashboard'))
        
        # Save to a temporary file
        temp_dir = os.path.join(app.root_path, 'tmp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = os.path.join(temp_dir, filename)
        
        # Determine file type and appropriate write mode
        is_json = False
        try:
            json.loads(content)
            is_json = True
        except (TypeError, json.JSONDecodeError):
            # Not JSON or not a string - check if it's HTML or plain text
            is_json = False
        
        # Write file in appropriate mode
        if is_json:
            with open(temp_file, 'w') as f:
                f.write(content)
        else:
            # If it appears to be HTML, save as HTML
            if isinstance(content, str) and ('<html' in content.lower() or '<!doctype html' in content.lower()):
                with open(temp_file, 'w') as f:
                    f.write(content)
            else:
                # Last resort - save as binary
                with open(temp_file, 'wb') as f:
                    f.write(content if isinstance(content, bytes) else str(content).encode('utf-8'))
        
        # Set proper filename extension
        download_filename = filename
        if is_json and not filename.endswith('.json'):
            download_filename = f"{filename}.json"
        elif '<html' in content.lower() and not filename.endswith(('.html', '.htm')):
            download_filename = f"{filename}.html"
        
        return send_from_directory(
            directory=temp_dir,
            filename=filename,
            as_attachment=True,
            download_name=download_filename
        )
    except Exception as e:
        app.logger.error(f"Error downloading report {report_id}: {e}", exc_info=True)
        flash(f'Error downloading report: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/stripe-webhook', methods=['POST'])
def stripe_webhook():
    # Handle Stripe webhook for payment confirmation
    # This would normally verify the signature and process the event
    payload = request.get_json()
    
    if payload.get('type') == 'checkout.session.completed':
        # Get the report ID from the metadata
        session = payload.get('data', {}).get('object', {})
        report_id = session.get('metadata', {}).get('report_id')
        
        if report_id:
            # Mark the report as paid
            mark_report_as_paid(report_id)
    
    return jsonify({"status": "success"}) 