import os
import json
import logging
import sys
import uuid # For guest session IDs
from datetime import datetime, timedelta
from urllib.parse import urlparse
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, abort, session, send_from_directory, g, current_app, Response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import re # Imported re for email validation in request_download_api
import argparse
import requests
from urllib.parse import quote, urlunparse
import urllib.parse

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('seo_agent')

# Create necessary directories
os.makedirs('results', exist_ok=True)
os.makedirs('tmp', exist_ok=True)
logger.info("Created necessary directories for file storage")

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

# Initialize Supabase and attach to app context
try:
    from supabase_config import get_supabase_client
    supabase_client = get_supabase_client()
    app.supabase_client = supabase_client # Attach for access in decorator via current_app
    logger.info("Connected to Supabase and attached to Flask app")
except Exception as e:
    logger.error(f"Failed to connect to Supabase: {e}")
    app.supabase_client = None # Ensure it exists even if None

# Initialize Celery
from celery_config import make_celery
celery = make_celery(app)

# Import tasks for Celery
from tasks import run_ai_powered_seo_analysis

# Import auth decorator and helper
from auth import token_required, get_current_user_id, try_get_user_from_token

# Initialize storage
try:
    from storage import init_storage
    # The init_storage function doesn't return anything, so we can't check a return value
    init_storage()
    
    # Run diagnostics to check if storage was initialized properly
    from storage import run_storage_diagnostics
    diagnostics = run_storage_diagnostics()
    
    # If we got this far without exceptions, consider it successful
    if diagnostics['status'] in ['all_checks_passed', 'partial_success']:
        logger.info(f"Storage initialized with status: {diagnostics['status']}")
    else:
        logger.warning(f"Storage initialized with status: {diagnostics['status']}. Some functions may not work correctly.")
        
    # Either way, storage is initialized enough to continue
    logger.info("Storage initialization completed without exceptions")
except Exception as e:
    logger.error(f"Exception during storage initialization: {e}", exc_info=True)
    logger.error("Storage operations will likely fail, affecting report storage and viewing.")

# Set up rate limiting with key_func as first parameter
limiter = Limiter(
    key_func=get_remote_address, # Consider if this should be user-based after auth
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"  # Use memory for rate limiting
)

@app.before_request
def load_user_if_token_present():
    # This runs before each request. If a token is present and valid, g.user will be set.
    # If no token or invalid, g.user will not be set (or explicitly set to None by try_get_user_from_token if it was called).
    # We call it here so g.user is available if needed, but routes can decide if it's mandatory.
    try_get_user_from_token()

@app.context_processor
def inject_current_year():
    return {'current_year': datetime.utcnow().year}

# Create a basic route
@app.route('/')
def index():
    return render_template('index.html')

# Add the analyze route, now protected
@app.route('/analyze', methods=['POST'])
@limiter.limit("10 per hour") # Rate limit can be user-specific later
def analyze():
    user_id = None
    guest_session_id = None

    if hasattr(g, 'user') and g.user:
        user_id = g.user.id
        app.logger.info(f"Analysis initiated by authenticated user: {user_id}")
    else:
        guest_session_id = session.get('guest_session_id')
        if not guest_session_id:
            guest_session_id = str(uuid.uuid4())
            session['guest_session_id'] = guest_session_id
            app.logger.info(f"New guest session for analysis: {guest_session_id}")
        else:
            app.logger.info(f"Analysis initiated by guest session: {guest_session_id}")
    
    url = request.form.get('url')
    if not url:
        flash('Please enter a valid URL', 'error')
        return redirect(url_for('index'))
    
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    pages = request.form.get('pages', 5, type=int)
    keyword = request.form.get('keyword', '')
    
    try:
        from data_access import create_analysis_job
        
        job = create_analysis_job(
            user_id=user_id, 
            guest_session_id=guest_session_id,
            url=url, 
            pages=pages, 
            keyword=keyword, 
            total_steps_initial=10 # Matches TOTAL_AI_STEPS in tasks.py
        )

        if not job or not job.get('filename'):
            flash('Error creating analysis job record', 'error')
            return redirect(url_for('index'))
        
        filename = job.get('filename')
        log_identifier = f"User: {user_id}" if user_id else f"Guest: {guest_session_id}"
        app.logger.info(f"Dispatching AI analysis for {url} ({log_identifier}, file: {filename})")
        
        run_ai_powered_seo_analysis.delay(
            url_to_analyze=url, 
            max_pages=pages, 
            user_id=user_id, 
            filename=filename, 
            keyword=keyword
        )
        
        return redirect(url_for('analysis_results_page', filename=filename)) # Renamed to avoid conflict
    except Exception as e:
        flash(f'Error starting analysis: {str(e)}', 'error')
        app.logger.error(f"Error in /analyze route: {str(e)}", exc_info=True)
        return redirect(url_for('index'))

# Results route, now protected
@app.route('/results/<filename>')
def analysis_results_page(filename): # Renamed from analysis_results to avoid conflict with API
    current_user_id = get_current_user_id()
    current_guest_session_id = session.get('guest_session_id')
    app.logger.info(f"ANALYSIS_RESULTS_PAGE: Loading page for filename: {filename}, User: {current_user_id}, Guest: {current_guest_session_id}")
    
    from data_access import get_analysis_job_by_filename
    # Explicitly use service role to bypass RLS and ensure we always get the data
    job = get_analysis_job_by_filename(filename, use_service_client_for_supabase=True)

    # Log the raw job data fetched
    if job:
        app.logger.info(f"ANALYSIS_RESULTS_PAGE: Job data fetched for template: {json.dumps(job, default=str)}")
    else:
        app.logger.warning(f"ANALYSIS_RESULTS_PAGE: No job data found for filename: {filename}. Rendering with job=None.")

    if not job:
        flash('Analysis not found or you may not have permission to view it.', 'error') # Modified flash message
        # Still render results.html but job will be None, template should handle this.
        return render_template('results.html', job=None, report_data_for_template=None)

    # Access Control (already present, good)
    job_owner_id = job.get('user_id')
    job_guest_id = job.get('guest_session_id')
    can_access = False
    if job_owner_id and current_user_id == job_owner_id:
        can_access = True
    elif job_guest_id and not job_owner_id and current_guest_session_id == job_guest_id:
        can_access = True

    if not can_access:
        flash('You do not have permission to view this analysis.', 'error')
        log_identifier = f"User {current_user_id}" if current_user_id else f"Guest session {current_guest_session_id}"
        app.logger.warning(f"ANALYSIS_RESULTS_PAGE: {log_identifier} attempted to access job {filename} without permission.")
        return redirect(url_for('index')) # Redirect if no access
    
    report_data_for_template = None
    if job.get('status') == 'completed':
        if job.get('storage_url'):
            report_data_for_template = {'type': 'html_url', 'url': job.get('storage_url')}
        elif job.get('raw_results'):
            report_data_for_template = {'type': 'json', 'data': job.get('raw_results')}
        # Add the full raw_results to report_data_for_template if it exists for completed jobs, for the preview
        if job.get('raw_results'):
            if report_data_for_template is None: report_data_for_template = {}
            report_data_for_template['raw_results_for_preview'] = job.get('raw_results')

    app.logger.info(f"ANALYSIS_RESULTS_PAGE: Rendering results.html for {filename} with job status: {job.get('status')}")
    return render_template(
        'results.html', 
        job=job, 
        report_data_for_template=report_data_for_template,
        current_year=datetime.utcnow().year  # Explicitly pass current_year
    )

# API endpoint for progress, now protected
@app.route('/api/progress/<filename>')
def get_progress(filename):
    current_user_id = get_current_user_id()
    current_guest_session_id = session.get('guest_session_id')

    from data_access import get_analysis_job_by_filename
    # Explicitly use service role to bypass RLS and ensure we always get the data
    job = get_analysis_job_by_filename(filename, use_service_client_for_supabase=True)
    
    if not job:
        return jsonify({"status": "not_found", "message": "Analysis job not found"}), 404

    job_owner_id = job.get('user_id')
    job_guest_id = job.get('guest_session_id')
    can_access = False
    if job_owner_id and current_user_id == job_owner_id:
        can_access = True
    elif job_guest_id and not job_owner_id and current_guest_session_id == job_guest_id:
        can_access = True

    if not can_access:
        log_identifier = f"User {current_user_id}" if current_user_id else f"Guest session {current_guest_session_id}"
        app.logger.warning(f"{log_identifier} unauthorized progress check for job {filename}.")
        return jsonify({"status": "unauthorized", "message": "You cannot view this job's progress"}), 403

    # Ensure status and progress are consistent
    job_status = job.get('status', 'pending')
    job_progress = job.get('progress', 0)
    
    # Make sure progress is a valid number
    try:
        job_progress = float(job_progress)
    except (ValueError, TypeError):
        job_progress = 0
    
    # Ensure status and progress are consistent
    if job_status == 'completed' and job_progress < 100:
        job_progress = 100
    elif job_status == 'running' and job_progress >= 100:
        job_progress = 99  # Cap at 99% if still running
    elif job_status == 'pending' and job_progress > 0:
        # If there's progress but status is pending, update to running
        job_status = 'running'
    
    # Update the job object with the consistent values
    job['status'] = job_status
    job['progress'] = job_progress

    if job.get('start_time') and isinstance(job.get('start_time'), str):
        try: job['start_time'] = datetime.fromisoformat(job['start_time']).timestamp()
        except ValueError: job['start_time'] = None
    
    if job.get('end_time') and isinstance(job.get('end_time'), str):
        try: job['end_time'] = datetime.fromisoformat(job['end_time']).timestamp()
        except ValueError: job['end_time'] = None
    
    return jsonify(job)

# API endpoint for results, now protected
@app.route('/api/results_data/<filename>') # Renamed from /api/results to avoid conflict
def get_results_data_api(filename):
    current_user_id = get_current_user_id()
    current_guest_session_id = session.get('guest_session_id')
    app.logger.info(f"API_RESULTS_DATA: Request for {filename}. User: {current_user_id}, Guest Session: {current_guest_session_id}")

    from data_access import get_analysis_job_by_filename
    # Explicitly use service role to bypass RLS and ensure data is always retrieved
    job = get_analysis_job_by_filename(filename, use_service_client_for_supabase=True)

    if not job:
        app.logger.warning(f"API_RESULTS_DATA: Job not found for {filename}.")
        return jsonify({"status": "not_found", "error": "Results file not found"}), 404
    
    app.logger.info(f"API_RESULTS_DATA: Job data fetched for {filename}: {json.dumps(job, default=str, indent=2)}") # Use default=str for datetime

    job_owner_id = job.get('user_id')
    job_guest_id = job.get('guest_session_id')
    can_access = False
    if job_owner_id and current_user_id == job_owner_id:
        can_access = True
    elif job_guest_id and not job_owner_id and current_guest_session_id == job_guest_id:
        can_access = True

    if not can_access:
        log_identifier = f"User {current_user_id}" if current_user_id else f"Guest session {current_guest_session_id}"
        app.logger.warning(f"API_RESULTS_DATA: {log_identifier} unauthorized results API access for job {filename}.")
        return jsonify({"status": "unauthorized", "error": "You cannot access these results"}), 403
        
    if job.get('status') in ['pending', 'running']:
        app.logger.info(f"API_RESULTS_DATA: Job {filename} is still {job.get('status')}.")
        return jsonify({"status": job.get('status'), "message": "Analysis in progress"}), 202
    
    if job.get('status') == 'failed':
        app.logger.warning(f"API_RESULTS_DATA: Job {filename} failed. Error: {job.get('error')}")
        return jsonify({"status": "failed", "error": job.get('error', "Unknown error")}), 500
    
    # If job is completed, construct the response payload
    final_response_payload = {"status": "completed"}

    if job.get('storage_url'):
        final_response_payload["report_type"] = "html_url"
        final_response_payload["url"] = job.get('storage_url')
        app.logger.info(f"API_RESULTS_DATA: Found storage_url for {filename}: {job.get('storage_url')}")
    
    if job.get('raw_results'):
        final_response_payload["raw_results"] = job.get('raw_results')
        # If only raw_results was found (no storage_url), set report_type to json
        if "report_type" not in final_response_payload: 
            final_response_payload["report_type"] = "json"
            # The JS might look for a 'data' key if type is json, mirroring old structure
            final_response_payload["data"] = job.get('raw_results') 
        app.logger.info(f"API_RESULTS_DATA: Including raw_results for {filename}.")

    # Check if we actually have content to send
    if "report_type" not in final_response_payload:
        app.logger.warning(f"API_RESULTS_DATA: Job {filename} completed but no storage_url or raw_results found for API response.")
        return jsonify({"status": "completed_no_data", "error": "Report data is not available in expected fields."}), 404
            
    app.logger.info(f"API_RESULTS_DATA: Sending response for {filename}: {json.dumps(final_response_payload, default=str, indent=2)}")
    return jsonify(final_response_payload)

# New API endpoint to get current user info
@app.route('/api/user')
@token_required
def get_user_info():
    if g.user:
        return jsonify({
            "id": g.user.id,
            "email": g.user.email,
        })
    return jsonify({"message": "User not found or error in token"}), 404

@app.route('/api/request_download/<filename>', methods=['POST'])
def request_download_api(filename):
    current_user_id = get_current_user_id()
    current_guest_session_id = session.get('guest_session_id')
    email = request.form.get('email')

    if not email: 
        return jsonify({"status": "error", "message": "Email is required for download."}), 400
    # Basic email validation (can be improved)
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return jsonify({"status": "error", "message": "Invalid email format."}), 400

    from data_access import get_analysis_job_by_filename, record_report_download
    job = get_analysis_job_by_filename(filename)

    if not job:
        return jsonify({"status": "error", "message": "Analysis job not found."}), 404

    job_owner_id = job.get('user_id')
    job_guest_id = job.get('guest_session_id')
    can_access = False
    if job_owner_id and current_user_id == job_owner_id:
        can_access = True
    elif job_guest_id and not job_owner_id and current_guest_session_id == job_guest_id:
        can_access = True
    
    if not can_access:
        log_identifier = f"User {current_user_id}" if current_user_id else f"Guest session {current_guest_session_id}"
        app.logger.warning(f"{log_identifier} unauthorized download request for job {filename}.")
        return jsonify({"status": "unauthorized", "message": "You cannot download this report."}), 403

    if job.get('status') != 'completed':
        return jsonify({"status": "error", "message": "Report is not yet complete."}) , 400
    
    storage_url_for_download = job.get('storage_url')
    if not storage_url_for_download:
         return jsonify({"status": "error", "message": "Report download URL not available."}), 404

    record_report_download(filename, email)
    
    return jsonify({"status": "success", "download_url": storage_url_for_download})

# Define a health check route
@app.route('/health')
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

# Add a route for storage diagnostics that can help troubleshoot storage issues
@app.route('/admin/storage-diagnostics')
def admin_storage_diagnostics():
    """Admin route to check storage system status."""
    from storage import run_storage_diagnostics, supabase, BUCKET_NAME
    
    # Run diagnostics
    diagnostics = run_storage_diagnostics()
    
    # List files in the bucket
    bucket_files = []
    public_url_examples = []
    
    if supabase:
        try:
            files = supabase.storage.from_(BUCKET_NAME).list()
            for file in files[:10]:  # Get first 10 files
                try:
                    # Handle different file structures
                    if hasattr(file, 'name'):
                        name = file.name
                    elif hasattr(file, 'get'):
                        name = file.get('name')
                    else:
                        name = str(file)
                    
                    file_info = {'name': name}
                    
                    # Get URLs for the first few files
                    if len(public_url_examples) < 3:
                        try:
                            # Try to get public URL
                            public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(name)
                            file_info['public_url'] = public_url
                            
                            # Check if URL needs fixing
                            if "/storage/v1/" in public_url and "/object/public" not in public_url:
                                parts = public_url.split("/storage/v1/")
                                if len(parts) == 2:
                                    base = parts[0]
                                    rest = parts[1]
                                    fixed_url = f"{base}/storage/v1/object/public/{rest}"
                                    file_info['fixed_url'] = fixed_url
                                    
                                    # Test both URLs
                                    import requests
                                    
                                    try:
                                        orig_response = requests.head(public_url, timeout=5)
                                        file_info['original_url_status'] = orig_response.status_code
                                    except Exception as e:
                                        file_info['original_url_error'] = str(e)
                                    
                                    try:
                                        fixed_response = requests.head(fixed_url, timeout=5)
                                        file_info['fixed_url_status'] = fixed_response.status_code
                                    except Exception as e:
                                        file_info['fixed_url_error'] = str(e)
                                    
                            # Add to examples
                            public_url_examples.append(file_info)
                        except Exception as e:
                            file_info['url_error'] = str(e)
                    
                    bucket_files.append(file_info)
                except Exception as e:
                    bucket_files.append({'error': str(e)})
        except Exception as e:
            bucket_files = [{'error': str(e)}]
    
    # Format output as HTML
    html_output = "<h1>Storage Diagnostics</h1>"
    html_output += "<h2>Diagnostic Results</h2>"
    html_output += "<pre>" + json.dumps(diagnostics, indent=2) + "</pre>"
    
    if bucket_files:
        html_output += "<h2>Bucket Files</h2>"
        html_output += "<pre>" + json.dumps(bucket_files, indent=2) + "</pre>"
    
    if public_url_examples:
        html_output += "<h2>URL Examples</h2>"
        html_output += "<pre>" + json.dumps(public_url_examples, indent=2) + "</pre>"
    
    html_output += "<h2>More Actions</h2>"
    html_output += "<ul>"
    html_output += "<li><a href='/admin/fix-supabase'>Fix Supabase Configuration</a></li>"
    html_output += "<li><a href='/admin/create-test-file'>Create Test File</a></li>"
    html_output += "<li><a href='/admin/list-bucket-files'>List Bucket Files</a></li>"
    html_output += "<li><a href='/admin/debug-bucket-objects'>Debug Bucket Objects</a></li>"
    
    # Add URL test link if we have examples
    if public_url_examples and len(public_url_examples) > 0:
        for example in public_url_examples:
            if 'fixed_url' in example:
                html_output += f"<li><a href='{example['fixed_url']}' target='_blank'>Test Fixed URL for {example['name']}</a></li>"
                break
            elif 'public_url' in example:
                html_output += f"<li><a href='{example['public_url']}' target='_blank'>Test Public URL for {example['name']}</a></li>"
                break
    
    html_output += "</ul>"
    
    html_output += """
    <h2>URL Fix Helper</h2>
    <p>If URLs are being generated incorrectly, use this form to fix them:</p>
    <form id="fixUrlForm">
        <div class="form-group">
            <label for="brokenUrl">Broken URL:</label>
            <input type="text" id="brokenUrl" class="form-control" style="width:100%; padding:8px; margin-bottom:10px;" 
                   placeholder="https://your-project.supabase.co/storage/v1/seo-reports/file.html">
        </div>
        <button type="button" id="fixUrlBtn" style="padding:8px 16px; background:#4CAF50; color:white; border:none; cursor:pointer;">Fix URL</button>
    </form>
    <div id="fixedUrlResult" style="margin-top:10px; padding:10px; border:1px solid #ddd; display:none;">
        <h3>Fixed URL:</h3>
        <p id="fixedUrl" style="word-break:break-all;"></p>
        <a id="fixedUrlLink" href="#" target="_blank">Test URL</a>
    </div>
    
    <script>
    document.getElementById('fixUrlBtn').addEventListener('click', function() {
        var brokenUrl = document.getElementById('brokenUrl').value;
        if (brokenUrl) {
            var fixedUrl = brokenUrl;
            if (brokenUrl.includes('/storage/v1/') && !brokenUrl.includes('/object/public/')) {
                var parts = brokenUrl.split('/storage/v1/');
                if (parts.length == 2) {
                    fixedUrl = parts[0] + '/storage/v1/object/public/' + parts[1];
                }
            }
            
            document.getElementById('fixedUrl').textContent = fixedUrl;
            document.getElementById('fixedUrlLink').href = fixedUrl;
            document.getElementById('fixedUrlResult').style.display = 'block';
        }
    });
    </script>
    """
    
    return html_output

# Helper route to create a test file in the bucket
@app.route('/admin/create-test-file')
def admin_create_test_file():
    from storage import save_report
    test_content = f"Test file created at {datetime.now().isoformat()}"
    test_filename = f"test_{int(datetime.now().timestamp())}.txt"
    
    try:
        result = save_report(test_content, test_filename, upload_to_storage=True)
        url = result.get('storage_url', f"/local_reports/{test_filename}")
        
        html_output = "<h1>Test File Created</h1>"
        html_output += f"<p>Filename: {test_filename}</p>"
        html_output += f"<p>URL: <a href='{url}' target='_blank'>{url}</a></p>"
        html_output += "<p>Try accessing the URL to see if it works.</p>"
        html_output += "<p><a href='/admin/storage-diagnostics'>Back to Diagnostics</a></p>"
        
        return html_output
    except Exception as e:
        return f"<h1>Error Creating Test File</h1><p>Error: {str(e)}</p><p><a href='/admin/storage-diagnostics'>Back to Diagnostics</a></p>"

# Helper route to list files in the bucket
@app.route('/admin/list-bucket-files')
def admin_list_bucket_files():
    from storage import BUCKET_NAME, supabase
    
    try:
        files = supabase.storage.from_(BUCKET_NAME).list()
        
        html_output = f"<h1>Files in Bucket: {BUCKET_NAME}</h1>"
        
        if not files:
            html_output += "<p>No files found in bucket.</p>"
        else:
            html_output += "<table border='1' style='border-collapse: collapse; width: 100%;'>"
            html_output += "<tr><th>File Name</th><th>Size</th><th>Created At</th><th>Actions</th></tr>"
            
            for file in files:
                # Handle different object types for file metadata
                try:
                    # Try dictionary access
                    file_name = file.get("name")
                    file_size = file.get("metadata", {}).get("size", "Unknown")
                    created_at = file.get("created_at", "Unknown")
                except (AttributeError, TypeError):
                    # Try attribute access
                    try:
                        file_name = file.name
                        file_size = getattr(file, 'metadata', {}).get('size', "Unknown") 
                        created_at = getattr(file, 'created_at', "Unknown")
                    except AttributeError:
                        # Fallback - just use string representation
                        file_name = str(file)
                        file_size = "Unknown"
                        created_at = "Unknown"
                        # Add debug output
                        logger.info(f"File object debug: {str(type(file))}: {file}")
                
                # Get both public and signed URLs
                try:
                    public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(file_name)
                    
                    # Fix URL format if needed (missing /object/public)
                    fixed_url = None
                    if "/storage/v1/" in public_url and "/object/public" not in public_url:
                        parts = public_url.split("/storage/v1/")
                        if len(parts) == 2:
                            base = parts[0]
                            rest = parts[1]
                            fixed_url = f"{base}/storage/v1/object/public/{rest}"
                    
                    # Get signed URL as fallback
                    signed_url = supabase.storage.from_(BUCKET_NAME).create_signed_url(file_name, 60*60)  # 1 hour
                    
                    # Create action links
                    actions = []
                    actions.append(f"<a href='{public_url}' target='_blank'>Public URL</a>")
                    actions.append(f"<a href='/admin/test-url?url={urllib.parse.quote(public_url)}'>Test Public URL</a>")
                    
                    if fixed_url:
                        actions.append(f"<a href='{fixed_url}' target='_blank'>Fixed URL</a>")
                        actions.append(f"<a href='/admin/test-url?url={urllib.parse.quote(fixed_url)}'>Test Fixed URL</a>")
                    
                    actions.append(f"<a href='{signed_url}' target='_blank'>Signed URL</a>")
                    actions.append(f"<a href='/admin/test-url?url={urllib.parse.quote(signed_url)}'>Test Signed URL</a>")
                    
                    # Add local fallback option
                    actions.append(f"<a href='/local_reports/{file_name}' target='_blank'>Local Fallback</a>")
                    
                    html_output += f"<tr><td>{file_name}</td><td>{file_size}</td><td>{created_at}</td><td>{' | '.join(actions)}</td></tr>"
                except Exception as e:
                    html_output += f"<tr><td>{file_name}</td><td>{file_size}</td><td>{created_at}</td><td>Error: {str(e)}</td></tr>"
            
            html_output += "</table>"
        
        html_output += "<p><a href='/admin/storage-diagnostics'>Back to Diagnostics</a></p>"
        return html_output
    except Exception as e:
        logger.error(f"Error listing bucket files: {e}", exc_info=True)
        return f"<h1>Error Listing Files</h1><p>Error: {str(e)}</p><p><a href='/admin/storage-diagnostics'>Back to Diagnostics</a></p>"

# Add a URL testing endpoint
@app.route('/admin/test-url')
def admin_test_url():
    url = request.args.get('url')
    if not url:
        return "<h1>Error</h1><p>No URL provided. Use ?url=https://example.com</p>"
    
    # Parse and normalize the URL
    try:
        parsed = urlparse(url)
        # Ensure path is properly encoded
        fixed_path = quote(parsed.path)
        normalized_url = urlunparse((parsed.scheme, parsed.netloc, fixed_path, parsed.params, parsed.query, parsed.fragment))
    except Exception as e:
        return f"<h1>URL Parse Error</h1><p>Error: {str(e)}</p>"
    
    # Test variations of the URL to find what works
    variations = []
    
    # Original URL
    variations.append(("Original URL", url))
    
    # Normalized URL if different
    if normalized_url != url:
        variations.append(("Normalized URL", normalized_url))
    
    # If URL contains '/storage/v1/' and doesn't have '/object/public', create a fixed version
    if "/storage/v1/" in url and "/object/public" not in url:
        parts = url.split("/storage/v1/")
        if len(parts) == 2:
            base = parts[0]
            rest = parts[1]
            fixed_url = f"{base}/storage/v1/object/public/{rest}"
            variations.append(("Fixed URL (added /object/public)", fixed_url))
    
    # Results collection
    results = []
    
    for label, test_url in variations:
        try:
            # Test HEAD request
            head_response = requests.head(test_url, timeout=5)
            head_status = head_response.status_code
            head_headers = dict(head_response.headers)
            
            # Test GET request
            get_response = requests.get(test_url, timeout=5)
            get_status = get_response.status_code
            content_type = get_response.headers.get('Content-Type')
            content_snippet = get_response.text[:100] + '...' if len(get_response.text) > 100 else get_response.text
            
            results.append({
                "label": label,
                "url": test_url,
                "head_status": head_status,
                "get_status": get_status,
                "content_type": content_type,
                "content_snippet": content_snippet,
                "response_headers": head_headers
            })
        except Exception as e:
            results.append({
                "label": label,
                "url": test_url,
                "error": str(e)
            })
    
    # Generate HTML response
    html_output = "<h1>URL Test Results</h1>"
    
    for result in results:
        html_output += f"<h2>{result['label']}</h2>"
        html_output += f"<p>URL: <a href='{result['url']}' target='_blank'>{result['url']}</a></p>"
        
        if "error" in result:
            html_output += f"<p style='color: red;'>Error: {result['error']}</p>"
        else:
            status_color = "green" if result["head_status"] in (200, 201, 202, 203, 204) else "red"
            html_output += f"<p>HEAD Status: <span style='color: {status_color};'>{result['head_status']}</span></p>"
            
            status_color = "green" if result["get_status"] in (200, 201, 202, 203, 204) else "red"
            html_output += f"<p>GET Status: <span style='color: {status_color};'>{result['get_status']}</span></p>"
            
            html_output += f"<p>Content Type: {result['content_type']}</p>"
            html_output += "<h3>Content Snippet:</h3>"
            html_output += f"<pre>{result['content_snippet']}</pre>"
            
            html_output += "<h3>Response Headers:</h3>"
            html_output += "<ul>"
            for key, value in result["response_headers"].items():
                html_output += f"<li><strong>{key}:</strong> {value}</li>"
            html_output += "</ul>"
    
    html_output += "<p><a href='/admin/storage-diagnostics'>Back to Diagnostics</a></p>"
    
    return html_output

# Add direct bucket debugging endpoint
@app.route('/admin/debug-bucket-objects')
def admin_debug_bucket_objects():
    """Debug endpoint to understand Supabase bucket object structure"""
    from storage import supabase, BUCKET_NAME
    
    html_output = "<h1>Supabase Bucket Objects Debug</h1>"
    html_output += "<h2>Library Info</h2>"
    
    # Get library version info
    import pkg_resources
    supabase_version = "Unknown"
    try:
        supabase_version = pkg_resources.get_distribution("supabase").version
    except:
        pass
    
    html_output += f"<p>Supabase Python library version: {supabase_version}</p>"
    
    # Get buckets and examine their structure
    try:
        buckets = supabase.storage.list_buckets()
        
        html_output += "<h2>Bucket Collection Debug</h2>"
        html_output += f"<p>Type of collection: {type(buckets)}</p>"
        html_output += f"<p>Number of buckets: {len(buckets)}</p>"
        
        if len(buckets) > 0:
            html_output += "<h2>First Bucket Object Analysis</h2>"
            
            bucket = buckets[0]
            html_output += f"<p>Type: {type(bucket)}</p>"
            
            # Try different access methods
            html_output += "<h3>Dictionary Access Attempts</h3>"
            html_output += "<ul>"
            for key in ["name", "id", "created_at", "updated_at"]:
                try:
                    value = bucket.get(key)
                    html_output += f"<li>bucket.get('{key}') = {value}</li>"
                except Exception as e:
                    html_output += f"<li>bucket.get('{key}') = Error: {str(e)}</li>"
            html_output += "</ul>"
            
            # Try attribute access
            html_output += "<h3>Attribute Access Attempts</h3>"
            html_output += "<ul>"
            for attr in ["name", "id", "created_at", "updated_at"]:
                try:
                    value = getattr(bucket, attr)
                    html_output += f"<li>bucket.{attr} = {value}</li>"
                except Exception as e:
                    html_output += f"<li>bucket.{attr} = Error: {str(e)}</li>"
            html_output += "</ul>"
            
            # Introspect the object
            html_output += "<h3>Object Introspection</h3>"
            html_output += "<ul>"
            
            # Check dir
            try:
                dir_attrs = dir(bucket)
                html_output += f"<li>dir(bucket) = {dir_attrs}</li>"
            except Exception as e:
                html_output += f"<li>dir(bucket) = Error: {str(e)}</li>"
                
            # Check methods you might call on bucket
            for method in ["to_dict", "dict", "__dict__", "__str__"]:
                try:
                    if method == "__dict__":
                        value = bucket.__dict__
                    elif method == "__str__":
                        value = bucket.__str__()
                    else:
                        value = getattr(bucket, method)()
                    html_output += f"<li>bucket.{method}() = {value}</li>"
                except Exception as e:
                    html_output += f"<li>bucket.{method}() = Error: {str(e)}</li>"
            
            # Get string representation
            html_output += f"<li>str(bucket) = {str(bucket)}</li>"
            html_output += f"<li>repr(bucket) = {repr(bucket)}</li>"
            
            html_output += "</ul>"
    except Exception as e:
        html_output += f"<p style='color: red;'>Error listing buckets: {str(e)}</p>"
    
    # Test URL formation
    html_output += "<h2>URL Formation Tests</h2>"
    
    # Create a test file
    import time
    from io import BytesIO
    test_filename = f"url_test_{int(time.time())}.txt"
    test_content = f"URL test file created at {datetime.now().isoformat()}"
    
    try:
        # Upload test file
        html_output += f"<p>Uploading test file '{test_filename}'...</p>"
        
        file_obj = BytesIO(test_content.encode('utf-8'))
        supabase.storage.from_(BUCKET_NAME).upload(
            file=file_obj,
            path=test_filename,
            file_options={"content-type": "text/plain"}
        )
        
        html_output += "<p style='color:green;'>✓ Test file uploaded successfully</p>"
        
        # Test various URL formation methods
        html_output += "<h3>URL Tests</h3>"
        html_output += "<table border='1' style='border-collapse: collapse; width: 100%;'>"
        html_output += "<tr><th>Method</th><th>Result</th><th>URL</th></tr>"
        
        # Test public URL
        try:
            public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(test_filename)
            html_output += f"<tr><td>get_public_url</td><td style='color:green;'>SUCCESS</td><td>{public_url}</td></tr>"
            
            # Test if URL format appears to be wrong
            if "/storage/v1/" in public_url and "/object/public" not in public_url:
                # Try to fix the URL
                parts = public_url.split("/storage/v1/")
                if len(parts) == 2:
                    base = parts[0]
                    rest = parts[1]
                    fixed_url = f"{base}/storage/v1/object/public/{rest}"
                    html_output += f"<tr><td>Fixed Public URL</td><td style='color:blue;'>CORRECTED</td><td>{fixed_url}</td></tr>"
                    
                    # Try to fetch with the fixed URL
                    try:
                        import requests
                        response = requests.head(fixed_url, timeout=5)
                        status = response.status_code
                        status_color = "green" if status in (200, 201, 202, 203, 204) else "red"
                        html_output += f"<tr><td>Fixed URL Test</td><td style='color:{status_color};'>{status}</td><td>HEAD request to fixed URL</td></tr>"
                    except Exception as e:
                        html_output += f"<tr><td>Fixed URL Test</td><td style='color:red;'>ERROR</td><td>{str(e)}</td></tr>"
        except Exception as e:
            html_output += f"<tr><td>get_public_url</td><td style='color:red;'>ERROR</td><td>{str(e)}</td></tr>"
        
        # Test signed URL
        try:
            signed_url = supabase.storage.from_(BUCKET_NAME).create_signed_url(test_filename, 60)
            html_output += f"<tr><td>create_signed_url</td><td style='color:green;'>SUCCESS</td><td>{signed_url}</td></tr>"
        except Exception as e:
            html_output += f"<tr><td>create_signed_url</td><td style='color:red;'>ERROR</td><td>{str(e)}</td></tr>"
            
        # Test direct download
        try:
            download_response = supabase.storage.from_(BUCKET_NAME).download(test_filename)
            content = download_response.decode('utf-8')
            html_output += f"<tr><td>download</td><td style='color:green;'>SUCCESS</td><td>Downloaded {len(content)} bytes</td></tr>"
        except Exception as e:
            html_output += f"<tr><td>download</td><td style='color:red;'>ERROR</td><td>{str(e)}</td></tr>"
            
        html_output += "</table>"
        
        # Clean up test file
        try:
            supabase.storage.from_(BUCKET_NAME).remove([test_filename])
            html_output += "<p>Test file was removed after tests.</p>"
        except Exception as e:
            html_output += f"<p>Warning: Could not remove test file: {str(e)}</p>"
        
    except Exception as e:
        html_output += f"<p style='color: red;'>Error during URL tests: {str(e)}</p>"
    
    html_output += "<hr><p><a href='/admin/storage-diagnostics'>Back to Storage Diagnostics</a></p>"
    
    return html_output

# Add a route to test and fix Supabase configuration
@app.route('/admin/fix-supabase')
def admin_fix_supabase_config():
    """Admin route to test, diagnose and fix Supabase configuration."""
    from supabase_config import test_supabase_connection
    
    # Run full Supabase tests
    connection_info = test_supabase_connection()
    
    # Force storage initialization
    from storage import init_storage
    try:
        init_storage()
    except Exception as e:
        connection_info["storage_init_error"] = str(e)
    
    # Attempt to update the bucket settings instead of recreating it
    try:
        from storage import supabase, BUCKET_NAME
        
        # Add bucket update result
        connection_info["bucket_settings"] = {}
        
        # Update bucket to be public
        try:
            bucket_update_result = supabase.storage.update_bucket(
                BUCKET_NAME, 
                options={
                    'public': True,
                    'file_size_limit': 52428800  # 50MB
                }
            )
            connection_info["bucket_settings"]["update_result"] = "success"
        except Exception as e:
            connection_info["bucket_settings"]["update_error"] = str(e)
        
        # List files in bucket
        try:
            files = supabase.storage.from_(BUCKET_NAME).list()
            connection_info["bucket_settings"]["file_count"] = len(files)
            
            # List first 5 files for debugging
            file_list = []
            for file in files[:5]:
                try:
                    if hasattr(file, 'name'):
                        name = file.name
                    elif hasattr(file, 'get'):
                        name = file.get('name')
                    else:
                        name = str(file)
                    file_list.append(name)
                except Exception as file_err:
                    file_list.append(f"Error getting file info: {file_err}")
            
            connection_info["bucket_settings"]["sample_files"] = file_list
        except Exception as e:
            connection_info["bucket_settings"]["list_error"] = str(e)
        
        # Upload a test file and get URL
        try:
            from io import BytesIO
            from datetime import datetime
            
            test_content = f"Test file created at {datetime.now().isoformat()}"
            test_file = BytesIO(test_content.encode('utf-8'))
            test_filename = "bucket_fix_test.txt"
            
            # Upload test file
            supabase.storage.from_(BUCKET_NAME).upload(
                file=test_file,
                path=test_filename,
                file_options={"content-type": "text/plain"}
            )
            connection_info["bucket_settings"]["upload_result"] = "success"
            
            # Get URL
            try:
                public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(test_filename)
                connection_info["bucket_settings"]["public_url"] = public_url
                
                # Check if URL needs fixing
                if "/storage/v1/" in public_url and "/object/public" not in public_url:
                    parts = public_url.split("/storage/v1/")
                    if len(parts) == 2:
                        base = parts[0]
                        rest = parts[1]
                        fixed_url = f"{base}/storage/v1/object/public/{rest}"
                        connection_info["bucket_settings"]["fixed_url"] = fixed_url
                        
                        # Test both URLs
                        import requests
                        
                        try:
                            orig_response = requests.head(public_url, timeout=5)
                            connection_info["bucket_settings"]["original_url_status"] = orig_response.status_code
                        except Exception as e:
                            connection_info["bucket_settings"]["original_url_error"] = str(e)
                        
                        try:
                            fixed_response = requests.head(fixed_url, timeout=5)
                            connection_info["bucket_settings"]["fixed_url_status"] = fixed_response.status_code
                        except Exception as e:
                            connection_info["bucket_settings"]["fixed_url_error"] = str(e)
            except Exception as e:
                connection_info["bucket_settings"]["get_url_error"] = str(e)
            
            # Clean up
            try:
                supabase.storage.from_(BUCKET_NAME).remove([test_filename])
                connection_info["bucket_settings"]["cleanup_result"] = "success"
            except Exception as e:
                connection_info["bucket_settings"]["cleanup_error"] = str(e)
                
        except Exception as e:
            connection_info["bucket_settings"]["test_file_error"] = str(e)
            
    except Exception as e:
        connection_info["bucket_settings_error"] = str(e)
    
    # Run diagnostics after fixes
    from storage import run_storage_diagnostics
    try:
        diagnostics = run_storage_diagnostics()
        connection_info["final_diagnostics"] = diagnostics
    except Exception as e:
        connection_info["final_diagnostics_error"] = str(e)
    
    # Add a helper for fixing URLs
    html_actions = """
    <h2>URL Fix Helper</h2>
    <p>If URLs are being generated incorrectly, use this form to fix them:</p>
    <form id="fixUrlForm">
        <div class="form-group">
            <label for="brokenUrl">Broken URL:</label>
            <input type="text" id="brokenUrl" class="form-control" style="width:100%; padding:8px; margin-bottom:10px;" 
                   placeholder="https://your-project.supabase.co/storage/v1/seo-reports/file.html">
        </div>
        <button type="button" id="fixUrlBtn" style="padding:8px 16px; background:#4CAF50; color:white; border:none; cursor:pointer;">Fix URL</button>
    </form>
    <div id="fixedUrlResult" style="margin-top:10px; padding:10px; border:1px solid #ddd; display:none;">
        <h3>Fixed URL:</h3>
        <p id="fixedUrl" style="word-break:break-all;"></p>
        <a id="fixedUrlLink" href="#" target="_blank">Test URL</a>
    </div>
    
    <script>
    document.getElementById('fixUrlBtn').addEventListener('click', function() {
        var brokenUrl = document.getElementById('brokenUrl').value;
        if (brokenUrl) {
            var fixedUrl = brokenUrl;
            if (brokenUrl.includes('/storage/v1/') && !brokenUrl.includes('/object/public/')) {
                var parts = brokenUrl.split('/storage/v1/');
                if (parts.length == 2) {
                    fixedUrl = parts[0] + '/storage/v1/object/public/' + parts[1];
                }
            }
            
            document.getElementById('fixedUrl').textContent = fixedUrl;
            document.getElementById('fixedUrlLink').href = fixedUrl;
            document.getElementById('fixedUrlResult').style.display = 'block';
        }
    });
    </script>
    """
    
    # Format as HTML for browser display
    html_output = "<h1>Supabase Configuration Fix</h1>"
    html_output += "<h2>Connection Test Results</h2>"
    html_output += "<pre>" + json.dumps(connection_info, indent=2) + "</pre>"
    
    html_output += html_actions
    
    html_output += "<h2>Actions</h2>"
    html_output += "<ul>"
    html_output += "<li><a href='/admin/storage-diagnostics'>View Storage Diagnostics</a></li>"
    html_output += "<li><a href='/admin/debug-bucket-objects'>Debug Bucket Objects</a></li>"
    html_output += "<li><a href='/admin/create-test-file'>Create Test File</a></li>"
    html_output += "</ul>"
    
    return html_output

# Add a route to serve local report files as fallback when storage fails
@app.route('/local_reports/<filename>')
def serve_local_report(filename):
    current_user_id = get_current_user_id()
    current_guest_session_id = session.get('guest_session_id')
    app.logger.info(f"LOCAL_REPORTS: Request for {filename}. User: {current_user_id}, Guest: {current_guest_session_id}")
    
    # Security check: Verify this is a valid report file
    from data_access import get_analysis_job_by_filename
    
    # Extract the base filename without extension
    base_filename = filename
    if filename.endswith('.html'):
        base_filename = filename[:-5]  # Remove .html
    elif filename.endswith('.json'):
        base_filename = filename[:-5]  # Remove .json
    elif filename.endswith('.css'):
        base_filename = filename[:-4]  # Remove .css
    elif filename.endswith('.js'):
        base_filename = filename[:-3]  # Remove .js
    
    # Get the job to check permissions
    job = get_analysis_job_by_filename(base_filename, use_service_client_for_supabase=True)
    
    if not job:
        app.logger.warning(f"LOCAL_REPORTS: No job found for {filename}")
        return jsonify({"error": "Report not found"}), 404
    
    # Check permissions
    job_owner_id = job.get('user_id')
    job_guest_id = job.get('guest_session_id')
    can_access = False
    if job_owner_id and current_user_id == job_owner_id:
        can_access = True
    elif job_guest_id and not job_owner_id and current_guest_session_id == job_guest_id:
        can_access = True
    
    if not can_access:
        log_identifier = f"User {current_user_id}" if current_user_id else f"Guest session {current_guest_session_id}"
        app.logger.warning(f"LOCAL_REPORTS: {log_identifier} attempted to access file {filename} without permission.")
        return jsonify({"error": "You do not have permission to access this report"}), 403
    
    # Determine content type with charset
    content_type = 'text/plain; charset=utf-8'  # Default
    if filename.endswith('.html'):
        content_type = 'text/html; charset=utf-8'
    elif filename.endswith('.json'):
        content_type = 'application/json; charset=utf-8'
    elif filename.endswith('.css'):
        content_type = 'text/css; charset=utf-8'
    elif filename.endswith('.js'):
        content_type = 'application/javascript; charset=utf-8'
    
    # Full path to the file
    results_dir = os.path.join(os.getcwd(), 'results')
    file_path = os.path.join(results_dir, filename)
    
    try:
        if not os.path.exists(file_path):
            app.logger.error(f"LOCAL_REPORTS: File not found: {file_path}")
            return jsonify({"error": "Report file not found"}), 404
            
        # Read file content directly
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create response with all necessary headers
        response = Response(content)
        response.headers['Content-Type'] = content_type
        response.headers['Content-Disposition'] = 'inline'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'SAMEORIGIN'
        response.headers['Cache-Control'] = 'no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        
        app.logger.info(f"LOCAL_REPORTS: Successfully served {filename} with content-type: {content_type}")
        return response
        
    except Exception as e:
        app.logger.error(f"LOCAL_REPORTS: Error serving file {filename}: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error serving report: {str(e)}"}), 500

def get_url_content(url):
    """Fetch content from a URL, with fallback mechanisms for Supabase storage URLs."""
    try:
        # Check if this is a local URL, if so, read from file
        if url.startswith('/local_reports/'):
            from storage import LOCAL_REPORTS_PATH
            local_path = os.path.join(LOCAL_REPORTS_PATH, os.path.basename(url))
            if os.path.exists(local_path):
                with open(local_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                logger.error(f"Local file not found: {local_path}")
                raise FileNotFoundError(f"Local file not found: {local_path}")
        
        # For Supabase URLs, try multiple approaches
        if 'supabase' in url:
            # First try the URL as-is
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    return response.text
                logger.warning(f"URL returned status {response.status_code}: {url}")
            except Exception as e:
                logger.warning(f"Error fetching URL content: {e}")
            
            # Try to fix URL format if needed
            if "/storage/v1/" in url and "/object/public" not in url:
                parts = url.split("/storage/v1/")
                if len(parts) == 2:
                    base = parts[0]
                    rest = parts[1]
                    fixed_url = f"{base}/storage/v1/object/public/{rest}"
                    logger.info(f"Trying corrected URL: {fixed_url}")
                    try:
                        response = requests.get(fixed_url, timeout=10)
                        if response.status_code == 200:
                            return response.text
                        logger.warning(f"Corrected URL returned status {response.status_code}")
                    except Exception as e:
                        logger.warning(f"Error fetching corrected URL content: {e}")
            
            # Try to get the file directly from Supabase
            try:
                # Extract filename from URL
                from urllib.parse import urlparse, unquote
                parsed_url = urlparse(url)
                path_parts = unquote(parsed_url.path).split('/')
                filename = next((p for p in reversed(path_parts) if p), None)
                
                if filename:
                    # Try to access the file using the storage API
                    from storage import supabase, BUCKET_NAME
                    
                    # Try to get the file content directly (no URL)
                    try:
                        response = supabase.storage.from_(BUCKET_NAME).download(filename)
                        if response:
                            return response.decode('utf-8')
                    except Exception as direct_err:
                        logger.warning(f"Could not get file directly from storage: {direct_err}")
                    
                    # Try to get a signed URL
                    try:
                        signed_url = supabase.storage.from_(BUCKET_NAME).create_signed_url(filename, 60*5)  # 5 minutes
                        logger.info(f"Generated signed URL: {signed_url}")
                        signed_response = requests.get(signed_url, timeout=10)
                        if signed_response.status_code == 200:
                            return signed_response.text
                    except Exception as signed_err:
                        logger.warning(f"Could not create/use signed URL: {signed_err}")
                    
                    # Try local file as last resort
                    from storage import LOCAL_REPORTS_PATH
                    local_path = os.path.join(LOCAL_REPORTS_PATH, filename)
                    if os.path.exists(local_path):
                        with open(local_path, 'r', encoding='utf-8') as f:
                            logger.info(f"Using local file as fallback: {local_path}")
                            return f.read()
            except Exception as supabase_err:
                logger.warning(f"Error trying to access file via Supabase API: {supabase_err}")
        
        # Regular URL fetch
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.text
        else:
            logger.error(f"Failed to fetch content, status code: {response.status_code}, URL: {url}")
            return None
    except Exception as e:
        logger.error(f"Error fetching URL content: {e}", exc_info=True)
        return None

@app.route('/direct_view/<filename>')
def direct_view_report(filename):
    """Direct view for debugging report rendering issues."""
    app.logger.info(f"DIRECT_VIEW: Attempting to display {filename} directly")
    
    # Security check still applies
    current_user_id = get_current_user_id()
    current_guest_session_id = session.get('guest_session_id')
    
    # Extract the base filename without extension
    base_filename = filename
    
    # Handle multiple file extensions (like .json.html)
    if '_analysis.json.html' in filename:
        base_filename = filename.replace('_analysis.json.html', '')
    elif '_analysis.json' in filename:
        base_filename = filename.replace('_analysis.json', '')
    elif '.' in filename:
        base_filename = filename.rsplit('.', 1)[0]
    
    app.logger.info(f"DIRECT_VIEW: Extracted base filename: {base_filename} from {filename}")
    
    # Get the job to check permissions
    from data_access import get_analysis_job_by_filename
    job = get_analysis_job_by_filename(base_filename, use_service_client_for_supabase=True)
    
    if not job:
        app.logger.warning(f"DIRECT_VIEW: No job found for {filename} (base: {base_filename})")
        return f"<html><body><h1>Error: Report not found</h1><p>No job found for {filename}</p></body></html>"
    
    # Check permissions
    job_owner_id = job.get('user_id')
    job_guest_id = job.get('guest_session_id')
    can_access = False
    if job_owner_id and current_user_id == job_owner_id:
        can_access = True
    elif job_guest_id and not job_owner_id and current_guest_session_id == job_guest_id:
        can_access = True
    
    if not can_access:
        return f"<html><body><h1>Access Denied</h1><p>You don't have permission to view this report.</p></body></html>"
    
    # Add .html extension if not present
    if not filename.endswith('.html'):
        filename = f"{filename}.html"
    
    # Full path to the file
    results_dir = os.path.join(os.getcwd(), 'results')
    file_path = os.path.join(results_dir, filename)
    
    try:
        if not os.path.exists(file_path):
            app.logger.warning(f"DIRECT_VIEW: File not found: {file_path}")
            
            # Try with _analysis.json.html extension as fallback
            if not filename.endswith('_analysis.json.html'):
                alternative_filename = f"{base_filename}_analysis.json.html"
                alternative_path = os.path.join(results_dir, alternative_filename)
                
                if os.path.exists(alternative_path):
                    app.logger.info(f"DIRECT_VIEW: Found alternative file: {alternative_filename}")
                    file_path = alternative_path
                else:
                    return f"<html><body><h1>File Not Found</h1><p>Report file {filename} does not exist.</p></body></html>"
            else:
                return f"<html><body><h1>File Not Found</h1><p>Report file {filename} does not exist.</p></body></html>"
            
        # Read file content directly
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply CSP fix to the HTML content
        from storage import ensure_report_has_js_functions
        fixed_content = ensure_report_has_js_functions(content)
        
        # Return the fixed HTML content with proper headers
        response = Response(fixed_content)
        response.headers['Content-Type'] = 'text/html; charset=utf-8'
        
        # Add headers to ensure proper rendering
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'  # Prevent framing
        response.headers['Cache-Control'] = 'no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        
        app.logger.info(f"DIRECT_VIEW: Successfully serving {filename} with CSP fix applied")
        return response
        
    except Exception as e:
        error_message = str(e)
        app.logger.error(f"DIRECT_VIEW: Error serving file {filename}: {error_message}", exc_info=True)
        return f"<html><body><h1>Error</h1><p>Failed to load report: {error_message}</p></body></html>"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SEO Analysis Tool')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    args = parser.parse_args()
    
    try:
        app.run(debug=True, port=args.port)
    except OSError as e:
        if 'Address already in use' in str(e):
            logger.error(f"Port {args.port} is already in use. Try a different port.")
        else:
            logger.error(f"Error starting server: {e}") 