import os
import json
from datetime import datetime, timezone
import logging
from supabase_config import get_supabase_client

logger = logging.getLogger('seo_agent')

# Initialize Supabase client
supabase = get_supabase_client()

# User operations
def create_user(user_id, email, username=None, is_admin=False):
    """Create a new user in the database"""
    user_data = {
        'user_id': user_id,
        'email': email,
        'username': username,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'is_admin': is_admin
    }
    
    response = supabase.table('users').insert(user_data).execute()
    return response.data[0] if response.data else None

def get_user_by_id(user_id):
    """Get a user by their ID"""
    response = supabase.table('users').select('*').eq('user_id', user_id).execute()
    return response.data[0] if response.data else None

def get_user_by_email(email):
    """Get a user by their email"""
    response = supabase.table('users').select('*').eq('email', email).execute()
    return response.data[0] if response.data else None

def update_user(user_id, **kwargs):
    """Update user information"""
    response = supabase.table('users').update(kwargs).eq('user_id', user_id).execute()
    return response.data[0] if response.data else None

# Website operations
def create_website(user_id, url, name=None):
    """Create a new website for a user"""
    website_data = {
        'user_id': user_id,
        'url': url,
        'name': name or url,
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    
    response = supabase.table('websites').insert(website_data).execute()
    return response.data[0] if response.data else None

def get_websites_by_user(user_id):
    """Get all websites for a user"""
    response = supabase.table('websites').select('*').eq('user_id', user_id).execute()
    return response.data

def get_website_by_id(website_id):
    """Get a website by its ID"""
    response = supabase.table('websites').select('*').eq('id', website_id).execute()
    return response.data[0] if response.data else None

# Report operations
def create_report(user_id, website_id, title, filename, status='pending', is_paid=False):
    """Create a new report for a website"""
    report_data = {
        'user_id': user_id,
        'website_id': website_id,
        'title': title,
        'filename': filename,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': status,
        'is_paid': is_paid
    }
    
    response = supabase.table('reports').insert(report_data).execute()
    return response.data[0] if response.data else None

def get_reports_by_user(user_id, limit=10):
    """Get reports for a user"""
    response = supabase.table('reports').select('*').eq('user_id', user_id).order('created_at', desc=True).limit(limit).execute()
    return response.data

def get_report_by_id(report_id):
    """Get a report by its ID"""
    response = supabase.table('reports').select('*').eq('id', report_id).execute()
    return response.data[0] if response.data else None

def get_report_by_filename(filename):
    """Get a report by its filename"""
    response = supabase.table('reports').select('*').eq('filename', filename).execute()
    return response.data[0] if response.data else None

def update_report(report_id, **kwargs):
    """Update report information"""
    response = supabase.table('reports').update(kwargs).eq('id', report_id).execute()
    return response.data[0] if response.data else None

def mark_report_as_paid(report_id):
    """Mark a report as paid"""
    return update_report(report_id, is_paid=True)

# Analysis job operations
# The global `supabase` client here is initialized with the ANON key by default.
# For operations triggered by Celery that need to bypass RLS or ensure writes,
# we should use a client initialized with the SERVICE ROLE key.

def create_analysis_job(user_id, url, pages=5, keyword=None, filename=None, total_steps_initial=None, guest_session_id=None):
    """Create a new analysis job, supporting guest or authenticated users."""
    current_supabase_client = get_supabase_client() 
    try:
        if user_id is None and guest_session_id is None:
            raise ValueError("Either user_id or guest_session_id must be provided for an analysis job.")
        
        if user_id is not None and guest_session_id is not None:
            logger.warning(f"create_analysis_job called with both user_id ({user_id}) and guest_session_id ({guest_session_id}). Prioritizing user_id.")
            guest_session_id = None

        if not filename:
            domain = url.replace('http://', '').replace('https://', '').split('/')[0]
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"{domain}_{timestamp}_analysis.json"
        
        job_data = {
            'user_id': user_id, 
            'guest_session_id': guest_session_id, 
            'filename': filename,
            'url': url,
            'pages': pages,
            'keyword': keyword,
            'status': 'pending',
            'progress': 0,
            'current_step': 'initializing',
            'steps_completed': 0,
            'total_steps': total_steps_initial if total_steps_initial is not None else 10,
            'start_time': datetime.now(timezone.utc).isoformat(),
            'end_time': None,
            'error': None,
            'storage_url': None,
            'raw_results': None
        }
        
        results_dir = os.path.join(os.getcwd(), 'results')
        os.makedirs(results_dir, exist_ok=True)
        job_path = os.path.join(results_dir, filename)
        with open(job_path, 'w') as f:
            json.dump(job_data, f, indent=2)
        
        log_identifier = f"User: {user_id}" if user_id else f"Guest: {guest_session_id}"
        logger.info(f"Created local job file at {job_path} ({log_identifier})")
        
        try:
            response = current_supabase_client.table('analysis_jobs').insert(job_data).execute()
            if response.data and len(response.data) > 0:
                logger.info(f"Successfully inserted job into Supabase: ID {response.data[0].get('id')} ({log_identifier})")
            else:
                error_details = getattr(response, 'error', 'N/A')
                logger.error(f"CRITICAL: Supabase insert for job {filename} ({log_identifier}) FAILED TO RETURN DATA. Error: {error_details}. Full response: {response}")
        except Exception as e:
            logger.error(f"CRITICAL: Supabase insert EXCEPTION for job {filename} ({log_identifier}): {str(e)}", exc_info=True)
        
        return job_data
        
    except Exception as e:
        logger.error(f"Error creating analysis job {filename if filename else 'unknown_filename'}: {str(e)}", exc_info=True)
        return None

def get_analysis_job_by_filename(filename, use_service_client_for_supabase=True):
    """Get an analysis job. Prefers local file. For Supabase, uses service client by default."""
    # This function might be called by app.py (anon/user context) or tasks.py (service context)
    # We default to service client (True) to ensure data is always accessible regardless of RLS
    try:
        results_dir = os.path.join(os.getcwd(), 'results')
        job_path = os.path.join(results_dir, filename)
        
        if os.path.exists(job_path):
            try:
                with open(job_path, 'r') as f:
                    job_data = json.load(f)
                logger.info(f"Retrieved job from local file: {job_path}")
                return job_data
            except Exception as local_error:
                logger.error(f"Error reading local job file {job_path}: {str(local_error)}")
        
        try:
            client_to_use = get_supabase_client(use_service_role=use_service_client_for_supabase)
            response = client_to_use.table('analysis_jobs').select('*').eq('filename', filename).maybe_single().execute()
            if response.data:
                logger.info(f"Retrieved job {filename} from Supabase (service client: {use_service_client_for_supabase})")
                return response.data
        except Exception as e:
            logger.warning(f"Error retrieving job {filename} from Supabase (service client: {use_service_client_for_supabase}): {str(e)}")
        
        logger.warning(f"Job not found locally or in Supabase: {filename}")
        return None
        
    except Exception as e:
        logger.error(f"Error in get_analysis_job_by_filename for {filename}: {str(e)}", exc_info=True)
        return None

def get_analysis_jobs_by_user(user_id, limit=10):
    """Get analysis jobs for a user from Supabase."""
    # This is called by app.py in user context, so anon/auth key is fine.
    client_to_use = get_supabase_client()
    try:
        response = client_to_use.table('analysis_jobs').select('*').eq('user_id', user_id).order('start_time', desc=True).limit(limit).execute()
        return response.data
    except Exception as e:
        logger.error(f"Error fetching jobs for user {user_id} from Supabase: {str(e)}")
        return []

def update_analysis_job(filename, update_data, use_service_client_for_supabase=False):
    """Update an analysis job. For Supabase, can use service client if specified (e.g., for Celery tasks)."""
    # This function is called by update_job_progress and complete_analysis_job, 
    # which are called by Celery tasks. So, it often SHOULD use the service client.
    updated_job_data_from_local = None
    try:
        results_dir = os.path.join(os.getcwd(), 'results')
        job_path = os.path.join(results_dir, filename)
        
        current_data = {}
        if os.path.exists(job_path):
            try:
                with open(job_path, 'r') as f:
                    current_data = json.load(f)
            except Exception as local_read_error:
                logger.error(f"Error reading local job file {job_path} before update: {str(local_read_error)}")
        
        current_data.update(update_data)
        
        with open(job_path, 'w') as f:
            json.dump(current_data, f, indent=2)
        logger.info(f"Updated local job file at {job_path}")
        updated_job_data_from_local = current_data
        
    except Exception as local_write_error:
        logger.error(f"Critical error updating local job file {job_path}: {str(local_write_error)}")

    try:
        client_to_use = get_supabase_client(use_service_role=use_service_client_for_supabase)
        # Crucially, specify `returning="representation"` (supabase-py v1) or handle client v2 behavior for returning data.
        # For supabase-py v2.x (commonly used now), .execute() on update by default returns a ModelResponse with data if successful.
        # If RLS is an issue for the *select after update*, using service role helps.
        response = client_to_use.table('analysis_jobs').update(update_data).eq('filename', filename).execute()
        
        if response.data and len(response.data) > 0:
            logger.info(f"Successfully updated job {filename} in Supabase (service client: {use_service_client_for_supabase})")
        else:
            logger.warning(f"Supabase update for job {filename} (service client: {use_service_client_for_supabase}) returned no data or empty data array. Error: {getattr(response, 'error', 'N/A')}")
    except Exception as e:
        logger.warning(f"Supabase update error for job {filename} (service client: {use_service_client_for_supabase}): {str(e)}")
        
    return updated_job_data_from_local

# update_job_progress and complete_analysis_job are called by Celery tasks.
# They should ensure that update_analysis_job uses the service client.

def update_job_progress(filename, progress, current_step=None, steps_completed=None, total_steps=None):
    """Update the progress of an analysis job. Uses service client for Supabase."""
    update_data = {'progress': progress}
    if current_step is not None: update_data['current_step'] = current_step
    if steps_completed is not None: update_data['steps_completed'] = steps_completed
    if total_steps is not None: update_data['total_steps'] = total_steps
    return update_analysis_job(filename, update_data, use_service_client_for_supabase=True)

def complete_analysis_job(filename, error=None, storage_url=None, report_payload=None):
    """Mark an analysis job as complete or failed. Uses service client for Supabase."""
    # Get current total_steps using a potentially service client if called from Celery
    # However, get_analysis_job_by_filename defaults to anon client if not specified.
    # Let's ensure it uses service client for this internal backend operation.
    current_job_data = get_analysis_job_by_filename(filename, use_service_client_for_supabase=True)
    current_total_steps = 10 
    if current_job_data and current_job_data.get('total_steps') is not None:
        current_total_steps = current_job_data.get('total_steps')

    update_data = {
        'status': 'failed' if error else 'completed',
        'end_time': datetime.now(timezone.utc).isoformat(),
        'progress': 100 if not error else (current_job_data.get('progress') if current_job_data else 0),
        'error': str(error) if error else None,
        'storage_url': storage_url if storage_url else (current_job_data.get('storage_url') if current_job_data else None),
        'raw_results': report_payload if report_payload else (current_job_data.get('raw_results') if current_job_data else None)
    }
    
    if not error:
        update_data['steps_completed'] = current_total_steps
        update_data['current_step'] = 'Analysis completed'
    else:
        update_data['current_step'] = 'Analysis failed' 
        update_data['steps_completed'] = current_job_data.get('steps_completed', 0) if current_job_data else 0

    return update_analysis_job(filename, update_data, use_service_client_for_supabase=True)

def record_report_download(job_filename, email):
    """Records an email address associated with a report download."""
    # This function needs to use a service role client to bypass RLS
    client_to_use = get_supabase_client(use_service_role=True)
    try:
        download_data = {
            'job_filename': job_filename,
            'email': email,
            'downloaded_at': datetime.now(timezone.utc).isoformat()
        }
        response = client_to_use.table('report_downloads').insert(download_data).execute()
        if response.data and len(response.data) > 0:
            logger.info(f"Recorded download for job {job_filename} by email {email}: ID {response.data[0].get('id')}")
            return response.data[0]
        else:
            logger.warning(f"Failed to record download for job {job_filename}. Error: {getattr(response, 'error', 'N/A')}")
            return None
    except Exception as e:
        logger.error(f"Error recording report download for job {job_filename}: {str(e)}", exc_info=True)
        return None

# Task operations
def create_task(report_id, task_type, task_id=None):
    """Create a new task for a report"""
    task_data = {
        'report_id': report_id,
        'task_id': task_id,
        'type': task_type,
        'status': 'pending',
        'created_at': datetime.now(timezone.utc).isoformat()
    }
    
    response = supabase.table('tasks').insert(task_data).execute()
    return response.data[0] if response.data else None

def get_tasks_by_report(report_id):
    """Get tasks for a report"""
    response = supabase.table('tasks').select('*').eq('report_id', report_id).execute()
    return response.data

def update_task(task_id, **kwargs):
    """Update task information"""
    response = supabase.table('tasks').update(kwargs).eq('id', task_id).execute()
    return response.data[0] if response.data else None

def complete_task(task_id, error=None):
    """Mark a task as completed or failed"""
    now = datetime.now(timezone.utc).isoformat()
    
    if error:
        return update_task(task_id, status='failed', completed_at=now, error=error)
    else:
        return update_task(task_id, status='completed', completed_at=now) 