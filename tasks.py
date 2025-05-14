import os
import time
from datetime import datetime, timezone
import logging
import json
import threading
import asyncio # Added for running async tasks
from celery import shared_task
import requests # Import requests for a simple synchronous network test
from urllib.parse import urlparse

from storage import save_report
from data_access import update_job_progress, complete_analysis_job
from main import generate_seo_report
from seo_new import main_example as analyze_with_ai # Import the AI analysis function

logger = logging.getLogger('seo_agent')

# Direct method to save a report JSON file to disk
def save_file(filename, data):
    results_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    with open(filepath, 'w') as f:
        if isinstance(data, str):
            f.write(data)
        else:
            json.dump(data, f, indent=2)
    return filepath

@shared_task(bind=True)
def run_seo_analysis(self, url, pages=5, user_id=None, filename=None, keyword=None):
    """
    Celery task to run the (old) SEO analysis from main.py
    
    Args:
        url: The URL to analyze
        pages: Number of pages to analyze
        user_id: The ID of the user who started the analysis
        filename: The filename to save results to
        keyword: Primary keyword to analyze for density
    """
    task_id = self.request.id
    
    # Update job status to running
    update_job_progress(filename, 0, 'starting (legacy engine)', 0) # Clarified engine
    
    try:
        # Define a progress callback for the analysis
        def progress_callback(progress, step, steps_completed):
            update_job_progress(filename, progress, step, steps_completed)
        
        # Run the SEO analysis
        report = generate_seo_report(url, pages, progress_callback, keyword)
        
        if not report:
            logger.error(f"Legacy analysis failed for {url}")
            complete_analysis_job(filename, error="Legacy analysis failed")
            return {"status": "failed", "error": "Legacy analysis failed"}
        
        # Save results to Supabase storage
        # Assuming report is JSON serializable as before
        result = save_report(json.dumps(report), filename, upload_to_storage=True)
        storage_url = result.get('storage_url', f"/local_reports/{filename}")
        
        # Update job with completed status and storage URL
        update_job_progress(filename, 100, 'completed (legacy engine)', 5) # Clarified engine
        complete_analysis_job(filename, storage_url=storage_url) # Pass storage_url
        
        logger.info(f"Legacy analysis completed for {url}")
        return {
            "status": "completed", 
            "url": url, 
            "filename": filename,
            "storage_url": storage_url
        }
    
    except Exception as e:
        logger.error(f"Error in legacy analysis for {url}: {str(e)}")
        complete_analysis_job(filename, error=str(e))
        return {"status": "failed", "error": str(e)}

@shared_task(bind=True)
def run_ai_powered_seo_analysis(self, url_to_analyze, max_pages=5, user_id=None, filename=None, keyword=None): # Added keyword for consistency, though main_example may not use it directly
    """
    Celery task to run the new AI-powered SEO analysis from seo_new.py
    
    Args:
        url_to_analyze: The URL to analyze
        max_pages: Maximum number of pages to crawl
        user_id: The ID of the user who started the analysis (for job tracking)
        filename: The filename to associate with the job and report
        keyword: Primary keyword (passed for consistency, seo_new.py might use it differently or not at all for its core AI analysis)
    """
    task_id = self.request.id
    logger.info(f"CELERY TASK STARTED: {task_id} for {url_to_analyze} (Job: {filename})")

    # ===== VERY BASIC NETWORK DIAGNOSTIC =====
    diagnostic_target_url = "https://www.google.com" # A reliable, simple target
    gemini_api_host_for_dns_test = "generativelanguage.googleapis.com"
    target_site_host_for_dns_test = urlparse(url_to_analyze).netloc

    logger.info(f"CELERY_TASK_DIAGNOSTIC ({task_id}): Pinging {diagnostic_target_url} with requests...")
    try:
        response = requests.get(diagnostic_target_url, timeout=10)
        logger.info(f"CELERY_TASK_DIAGNOSTIC ({task_id}): Ping to {diagnostic_target_url} status: {response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"CELERY_TASK_DIAGNOSTIC ({task_id}): Ping to {diagnostic_target_url} FAILED: {e}")
    
    logger.info(f"CELERY_TASK_DIAGNOSTIC ({task_id}): Attempting DNS lookup for {gemini_api_host_for_dns_test}...")
    try:
        import socket
        ip_address = socket.gethostbyname(gemini_api_host_for_dns_test)
        logger.info(f"CELERY_TASK_DIAGNOSTIC ({task_id}): DNS lookup for {gemini_api_host_for_dns_test} SUCCEEDED: {ip_address}")
    except socket.gaierror as e:
        logger.error(f"CELERY_TASK_DIAGNOSTIC ({task_id}): DNS lookup for {gemini_api_host_for_dns_test} FAILED: {e}")

    if target_site_host_for_dns_test:
        logger.info(f"CELERY_TASK_DIAGNOSTIC ({task_id}): Attempting DNS lookup for target site host {target_site_host_for_dns_test}...")
        try:
            import socket
            ip_address_target = socket.gethostbyname(target_site_host_for_dns_test)
            logger.info(f"CELERY_TASK_DIAGNOSTIC ({task_id}): DNS lookup for {target_site_host_for_dns_test} SUCCEEDED: {ip_address_target}")
        except socket.gaierror as e:
            logger.error(f"CELERY_TASK_DIAGNOSTIC ({task_id}): DNS lookup for {target_site_host_for_dns_test} FAILED: {e}")
    # ===== END BASIC NETWORK DIAGNOSTIC =====

    # Update job status to running
    # Note: The progress reporting in main_example is (message_string, percentage_float)
    # We need to adapt this to update_job_progress(filename, progress_percentage, current_step_message, steps_completed_numeric)
    # steps_completed_numeric might be harder to map directly.
    
    # For simplicity, we'll map percentage_float to progress and message_string to current_step.
    # We might need a more sophisticated mapping for 'steps_completed' or adjust data_access.py later.
    # Let's assume a total of, say, 10 abstract steps for the AI analysis for now for progress display.
    TOTAL_AI_STEPS = 10 # Arbitrary number for now

    def progress_adapter_for_ai_analysis(step_message, percentage_progress):
        # Ensure percentage_progress is between 0 and 100
        progress_val = min(max(int(percentage_progress * 100), 0), 100)
        
        # Approximate steps_completed based on percentage
        steps_done = int(percentage_progress * TOTAL_AI_STEPS)
        
        logger.info(f"AI Analysis Progress for {filename}: {progress_val}% - {step_message} (Step {steps_done}/{TOTAL_AI_STEPS})")
        update_job_progress(filename, progress_val, step_message, steps_done, TOTAL_AI_STEPS)

    update_job_progress(filename, 0, 'Initializing AI analysis', 0, TOTAL_AI_STEPS)

    try:
        # Run the asynchronous main_example function from seo_new.py
        # Pass the adapter as the progress_callback
        analysis_result = asyncio.run(analyze_with_ai(
            url_to_analyze=url_to_analyze,
            max_pages=max_pages,
            # gemini_api_key_arg, dataforseo_login, dataforseo_password will be fetched from env by main_example
            progress_callback=progress_adapter_for_ai_analysis
            # architecture_recs can be added if needed, defaulting to None
        ))

        if not analysis_result or not analysis_result.get('report_path'):
            error_message = "AI analysis did not produce a report or failed."
            logger.error(f"{error_message} for {url_to_analyze} (Job: {filename})")
            complete_analysis_job(filename, error=error_message)
            return {"status": "failed", "error": error_message, "filename": filename}

        report_path = analysis_result.get('report_path') # This is a local path to the HTML report
        logger.info(f"AI analysis for {url_to_analyze} (Job: {filename}) generated report at: {report_path}")

        # TODO: Decide how to handle the 'report_path' from seo_new.py.
        # Option 1: Read the HTML file and save its content to Supabase storage.
        # Option 2: If Celery workers and Flask app share a filesystem (e.g., Docker volume),
        #           the path might be usable, but cloud storage is more robust for scaling.
        # For now, let's assume we want to upload the HTML report content to storage,
        # similar to the old task. We'll need to read the file content first.
        
        storage_url = None
        local_url = None
        
        try:
            if os.path.exists(report_path):
                with open(report_path, 'r', encoding='utf-8') as f:
                    html_report_content = f.read()
                
                # Use the job's filename for the storage object name
                html_filename = f"{filename}.html"
                
                # Always save locally first to guarantee a local fallback
                results_dir = os.path.join(os.getcwd(), 'static', 'reports')
                os.makedirs(results_dir, exist_ok=True)
                local_path = os.path.join(results_dir, html_filename)
                with open(local_path, 'w', encoding='utf-8') as f:
                    f.write(html_report_content)
                local_url = f"/local_reports/{html_filename}"
                logger.info(f"AI report for {filename} saved locally at: {local_path}")
                
                # Now try uploading to Supabase storage
                try:
                    result = save_report(html_report_content, html_filename, upload_to_storage=True)
                    storage_url = result.get('storage_url', f"/local_reports/{html_filename}")
                    logger.info(f"AI report for {filename} uploaded to storage: {storage_url}")
                    
                    # If storage_url is a local URL (indicating storage failed), use local_url
                    if storage_url and storage_url.startswith('/local_reports/'):
                        logger.warning(f"Supabase storage failed, using local URL for {filename}")
                        storage_url = local_url
                except Exception as e_storage:
                    logger.error(f"Error uploading AI report to storage: {str(e_storage)}")
                    # Fall back to local URL
                    storage_url = local_url
            else:
                logger.warning(f"AI generated report file not found at {report_path} for job {filename}. Cannot upload to storage.")
        except Exception as e_storage:
            logger.error(f"Error uploading AI report for {filename} from {report_path} to storage: {str(e_storage)}")
            # Continue to mark job as complete but without storage_url or with an error note about storage
        
        # Update job with completed status and storage URL
        update_job_progress(filename, 100, 'AI analysis completed', TOTAL_AI_STEPS, TOTAL_AI_STEPS)
        complete_analysis_job(filename, storage_url=storage_url, report_payload=analysis_result) # Optionally save full JSON result too
        
        logger.info(f"AI-powered analysis completed for {url_to_analyze} (Job: {filename})")
        return {
            "status": "completed",
            "url": url_to_analyze,
            "filename": filename,
            "storage_url": storage_url,
            "local_url": local_url,
            "details": analysis_result # Full analysis result from seo_new.py
        }
    
    except Exception as e:
        error_msg = f"Error in AI-powered analysis for {url_to_analyze} (Job: {filename}): {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        complete_analysis_job(filename, error=str(e))
        return {"status": "failed", "error": str(e), "filename": filename}


# Add a synchronous version of the analysis function
def run_analysis_sync(url, pages=5, user_id=None, filename=None, keyword=None):
    """
    Synchronous version of the analysis task for when Celery isn't running
    (This will be deprecated or removed once AI task is primary)
    
    Args:
        url: The URL to analyze
        pages: Number of pages to analyze
        user_id: The ID of the user who started the analysis
        filename: The filename to save results to
        keyword: Primary keyword to analyze for density
    """
    logger.info(f"Running synchronous analysis for {url} (LEGACY ENGINE)")
    
    try:
        # Get the job status file
        job_dir = os.path.join(os.getcwd(), 'results')
        os.makedirs(job_dir, exist_ok=True)
        job_file = os.path.join(job_dir, filename)
        
        # Load current job data
        try:
            with open(job_file, 'r') as f:
                job_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading job file: {str(e)}")
            job_data = {
                'user_id': user_id,
                'filename': filename,
                'url': url,
                'pages': pages,
                'keyword': keyword,
                'status': 'pending',
                'progress': 0,
                'current_step': 'initializing (legacy_sync)',
                'steps_completed': 0,
                'total_steps': 5, # Corresponds to main.py's steps
                'start_time': datetime.now(timezone.utc).isoformat()
            }
        
        # Update job status to running
        job_data['status'] = 'running'
        job_data['current_step'] = 'starting (legacy_sync)'
        job_data['progress'] = 0
        save_file(filename, job_data) # This saves to results/<filename> which is the job status file
        
        # Simple progress callback that directly updates the job file
        def progress_callback(progress, step, steps_completed):
            try:
                # This needs to read the JOB file, not the report file
                job_status_filepath = os.path.join(os.getcwd(), 'results', filename)
                with open(job_status_filepath, 'r') as f: # Read job status file
                    current_data = json.load(f)
                
                current_data['progress'] = progress
                current_data['current_step'] = step
                current_data['steps_completed'] = steps_completed
                
                save_file(filename, current_data) # Update job status file
                logger.info(f"Updated sync progress for {filename}: {progress}%, Step: {step}")
            except Exception as e:
                logger.error(f"Error updating sync progress for {filename}: {str(e)}")
        
        # Run the SEO analysis (from main.py)
        report = generate_seo_report(url, pages, progress_callback, keyword)
        
        if not report:
            logger.error(f"Legacy sync analysis failed for {url}")
            job_status_filepath = os.path.join(os.getcwd(), 'results', filename)
            with open(job_status_filepath, 'r') as f: # Read job status file
                current_data = json.load(f)
            
            current_data['status'] = 'failed'
            current_data['error'] = "Legacy sync analysis failed"
            current_data['end_time'] = datetime.now(timezone.utc).isoformat()
            
            save_file(filename, current_data) # Update job status file
            return {"status": "failed", "error": "Legacy sync analysis failed"}
        
        # Save the report (this is the actual analysis content, not the job status)
        # The `filename` for `save_file` here might be confusing.
        # Let's assume `run_analysis_sync` is for development and might save the report directly
        # with a name derived from `filename`, e.g., `filename_report.json`.
        # However, the current `app.py` expects the results/<filename> to BE the report.
        # The original `save_file(f"{filename}", report)` implies this.
        # This whole sync function's file handling is a bit mixed up with job status vs report data.
        # For now, we keep it as is, assuming `results/<filename>` becomes the report.
        
        report_content_path = save_file(filename, report) # This overwrites the job status file with the report.
        logger.info(f"Legacy sync report saved to: {report_content_path} (overwrote job status file)")
        
        # Update job with completed status (by re-creating a job status representation)
        # Since report_content_path IS results/<filename>
        final_job_status = {
            'user_id': user_id,
            'filename': filename,
            'url': url,
            'pages': pages,
            'keyword': keyword,
            'status': 'completed',
            'progress': 100,
            'current_step': 'completed (legacy_sync)',
            'steps_completed': 5, # from main.py
            'total_steps': 5,
            'start_time': job_data.get('start_time'), # Preserve start time
            'end_time': datetime.now(timezone.utc).isoformat(),
            'report_path_local': report_content_path 
            # In this sync mode, there's no separate storage_url unless we add that logic.
        }
        # We don't save this final_job_status anywhere explicitly as a separate file,
        # because `results/<filename>` is now the report itself.
        # The calling function in `app.py` reads `results/<filename>` as the report.
        
        logger.info(f"Legacy sync analysis completed for {url}")
        return {
            "status": "completed", 
            "url": url, 
            "filename": filename
            # "report_path_local": report_content_path # This is implicit by reading results/filename
        }
    
    except Exception as e:
        logger.error(f"Error in legacy sync analysis: {str(e)}")
        try:
            job_status_filepath = os.path.join(os.getcwd(), 'results', filename)
            # Try to update the job status file to failed
            with open(job_status_filepath, 'r') as f: # Read job status file (if it wasn't overwritten)
                current_data = json.load(f)
            
            current_data['status'] = 'failed'
            current_data['error'] = str(e)
            current_data['end_time'] = datetime.now(timezone.utc).isoformat()
            
            save_file(filename, current_data) # Update job status file
        except Exception as update_error:
            logger.error(f"Error updating legacy sync job status to failed for {filename}: {str(update_error)}")
        
        return {"status": "failed", "error": str(e)} 