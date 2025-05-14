import os
import json
import logging
from supabase_config import get_supabase_client
from io import BytesIO
from pathlib import Path
import traceback
import urllib.parse
import time
from datetime import datetime
try:
    # The exception is in the utils module not utils.exceptions
    from storage3.utils import StorageException
except ImportError:
    # Fallback to make sure there's always a StorageException class
    class StorageException(Exception):
        pass

logger = logging.getLogger('seo_agent')

# Initialize storage client with service role key for backend operations
try:
    supabase = get_supabase_client(use_service_role=True)
    logger.info("Supabase client for storage.py initialized with service role privileges.")
except ValueError as e:
    logger.error(f"Failed to initialize Supabase client in storage.py: {e}. Storage operations will likely fail.")
    supabase = None # Ensure supabase is defined, even if None, to avoid NameError later

BUCKET_NAME = 'seo-reports'
LOCAL_REPORTS_PATH = 'static/reports/'

def run_storage_diagnostics():
    """Run diagnostics on storage to help troubleshoot issues."""
    results = {
        "status": "initialized",
        "checks": {},
        "errors": {}
    }
    
    try:
        # Check if Supabase client is initialized
        results["checks"]["client_initialized"] = supabase is not None
        if not supabase:
            results["errors"]["client_init"] = "Supabase client failed to initialize"
            results["status"] = "client_init_failed"
            return results
        
        # Check bucket existence
        try:
            buckets = supabase.storage.list_buckets()
            
            # Handle different bucket object types (dict vs SyncBucket object)
            bucket_names = []
            results["bucket_objects_debug"] = []
            
            for bucket in buckets:
                bucket_info = {"original_type": str(type(bucket))}
                
                try:
                    # Try dictionary access
                    bucket_info["dict_access"] = str(bucket.get("name", "No name via dict"))
                    name = bucket.get("name")
                except (AttributeError, TypeError):
                    bucket_info["dict_access"] = "Failed"
                    try:
                        # Try attribute access
                        bucket_info["attr_access"] = str(bucket.name)
                        name = bucket.name
                    except AttributeError:
                        bucket_info["attr_access"] = "Failed"
                        # Try string representation
                        bucket_str = str(bucket)
                        bucket_info["string_repr"] = bucket_str[:100]  # Just a sample
                        if "name" in bucket_str:
                            import re
                            match = re.search(r"name='([^']+)'", bucket_str)
                            if match:
                                name = match.group(1)
                                bucket_info["regex_match"] = name
                            else:
                                name = None
                                bucket_info["regex_match"] = "Failed"
                        else:
                            name = None
                            bucket_info["string_search"] = "No name in string"
                
                results["bucket_objects_debug"].append(bucket_info)
                
                if name:
                    bucket_names.append(name)
            
            results["checks"]["buckets_listed"] = True
            results["data"] = {"available_buckets": bucket_names}
            
            # Check if our target bucket exists
            results["checks"]["target_bucket_exists"] = BUCKET_NAME in bucket_names
            if not results["checks"]["target_bucket_exists"]:
                results["errors"]["bucket_missing"] = f"Bucket '{BUCKET_NAME}' not found in available buckets"
                
                # Try to create the bucket if missing
                try:
                    supabase.storage.create_bucket(BUCKET_NAME, options={'public': True})
                    results["checks"]["bucket_created"] = True
                    results["checks"]["target_bucket_exists"] = True  # Update status after creation
                except Exception as e:
                    results["errors"]["bucket_creation"] = str(e)
                    results["checks"]["bucket_created"] = False
        except Exception as e:
            results["errors"]["bucket_list"] = str(e)
            results["checks"]["buckets_listed"] = False
            results["status"] = "bucket_list_failed"
            # Continue instead of returning, so we can still run URL tests
        
        # Test bucket access
        try:
            # Try to access the bucket
            bucket_info = supabase.storage.from_(BUCKET_NAME)
            results["checks"]["bucket_access"] = True
            
            # Try to list files
            try:
                files = bucket_info.list()
                results["checks"]["file_listing"] = True
                
                # Handle file objects which might also be different types
                files_info = []
                for file_obj in files:
                    try:
                        # Try dictionary access first
                        file_name = file_obj.get("name")
                    except (AttributeError, TypeError):
                        try:
                            # Try attribute access
                            file_name = file_obj.name
                        except AttributeError:
                            # Last resort - debug info
                            file_name = str(file_obj)
                            results["file_object_debug"] = str(type(file_obj))
                    
                    files_info.append(file_name)
                
                results["data"]["files_count"] = len(files)
                results["data"]["sample_files"] = files_info[:5] if files_info else []
            except Exception as e:
                results["errors"]["file_listing"] = str(e)
                results["checks"]["file_listing"] = False
        except Exception as e:
            results["errors"]["bucket_access"] = str(e)
            results["checks"]["bucket_access"] = False
            results["status"] = "bucket_access_failed"
            return results
        
        # Test URL formation
        try:
            test_file = "diagnostic_test.txt"
            
            # Test public URL formation
            try:
                public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(test_file)
                results["checks"]["public_url_generation"] = True
                results["data"]["public_url_example"] = public_url
                
                # Check URL format for common issues
                if "/storage/v1/" in public_url and "/object/public" not in public_url:
                    results["warnings"] = results.get("warnings", {})
                    results["warnings"]["url_format"] = "URL may be incorrectly formatted (missing '/object/public')"
                    
                    # Try to fix the URL for testing
                    parts = public_url.split("/storage/v1/")
                    if len(parts) == 2:
                        base = parts[0]
                        rest = parts[1]
                        fixed_url = f"{base}/storage/v1/object/public/{rest}"
                        results["data"]["corrected_url_example"] = fixed_url
            except Exception as e:
                results["errors"]["public_url_generation"] = str(e)
                results["checks"]["public_url_generation"] = False
            
            # Test signed URL formation - expected to fail if the file doesn't exist
            try:
                # Check if there are existing files we can use for a signed URL test
                try:
                    files = supabase.storage.from_(BUCKET_NAME).list()
                    if files and len(files) > 0:
                        # Try to get first existing file name
                        try:
                            first_file = files[0]
                            if hasattr(first_file, 'name'):
                                existing_file = first_file.name
                            elif hasattr(first_file, 'get'):
                                existing_file = first_file.get('name')
                            else:
                                # Extract from string representation
                                file_str = str(first_file)
                                import re
                                match = re.search(r"name='([^']+)'", file_str)
                                if match:
                                    existing_file = match.group(1)
                                else:
                                    existing_file = None
                            
                            if existing_file:
                                results["test_file_for_signed_url"] = existing_file
                                test_file = existing_file
                        except (IndexError, AttributeError, KeyError) as e:
                            results["file_access_error"] = str(e)
                except Exception as file_list_err:
                    results["file_list_error_for_signed_url"] = str(file_list_err)
                    
                # Try to create a signed URL - may fail if file doesn't exist
                signed_url = supabase.storage.from_(BUCKET_NAME).create_signed_url(test_file, 60)
                results["checks"]["signed_url_generation"] = True
                results["data"]["signed_url_example"] = signed_url
            except Exception as e:
                # This error is expected if file doesn't exist - don't treat as a failure
                if "not_found" in str(e) or "Object not found" in str(e):
                    results["checks"]["signed_url_generation"] = "skipped"
                    results["notes"] = results.get("notes", {})
                    results["notes"]["signed_url"] = f"Signed URL test skipped - test file '{test_file}' doesn't exist (this is normal)"
                    
                    # Try to create a temporary test file for signed URL test
                    try:
                        logger.info("Creating temporary test file for signed URL test")
                        test_content = f"Test file created at {datetime.now().isoformat()}"
                        temp_test_file = f"temp_test_{int(time.time())}.txt"
                        
                        # Use BytesIO for upload
                        temp_file_obj = BytesIO(test_content.encode('utf-8'))
                        
                        # Upload temporary file
                        supabase.storage.from_(BUCKET_NAME).upload(
                            file=temp_file_obj,
                            path=temp_test_file,
                            file_options={"content-type": "text/plain"}
                        )
                        
                        # Now try signed URL again with the temp file
                        signed_url = supabase.storage.from_(BUCKET_NAME).create_signed_url(temp_test_file, 60)
                        results["checks"]["signed_url_generation"] = True
                        results["data"]["signed_url_example"] = signed_url
                        results["notes"]["temp_file"] = f"Created temporary file '{temp_test_file}' for URL test"
                        
                        # Try to clean up the temporary file
                        try:
                            supabase.storage.from_(BUCKET_NAME).remove([temp_test_file])
                            results["notes"]["cleanup"] = f"Removed temporary test file '{temp_test_file}'"
                        except Exception as cleanup_err:
                            results["notes"]["cleanup_error"] = str(cleanup_err)
                    except Exception as temp_file_err:
                        results["notes"]["temp_file_error"] = str(temp_file_err)
                else:
                    # This is a different kind of error, so report it
                    results["errors"]["signed_url_generation"] = str(e)
                    results["checks"]["signed_url_generation"] = False
        except Exception as e:
            results["errors"]["url_tests"] = str(e)
            results["status"] = "url_tests_failed"
    except Exception as outer_e:
        # Catch-all exception handler to ensure we always return a valid result
        logger.error(f"Unexpected error in storage diagnostics: {outer_e}")
        results["errors"]["unexpected_error"] = str(outer_e)
        results["status"] = "failed"
        # Ensure critical fields exist even if there was an error
        if "checks" not in results:
            results["checks"] = {}
        if "data" not in results:
            results["data"] = {}
    
    # Final status determination - make sure this is always done before returning
    try:
        all_required_checks = all(
            results["checks"].get(check) is True 
            for check in ["buckets_listed", "target_bucket_exists", "bucket_access", "file_listing", "public_url_generation"]
        )
        
        # For signed URL, also consider "skipped" as a pass for this check
        signed_url_ok = results["checks"].get("signed_url_generation") is True or results["checks"].get("signed_url_generation") == "skipped"
        
        if all_required_checks and signed_url_ok:
            results["status"] = "all_checks_passed"
        elif results["checks"].get("target_bucket_exists", False) and results["checks"].get("bucket_access", False):
            results["status"] = "partial_success"
        else:
            results["status"] = "some_checks_failed"
    except Exception as status_e:
        logger.error(f"Error determining diagnostics status: {status_e}")
        results["errors"]["status_error"] = str(status_e)
        results["status"] = "unknown"
    
    return results

def init_storage():
    """Initialize storage system - handles both local and Supabase storage."""
    global supabase
    
    # Ensure local storage directory exists
    os.makedirs('results', exist_ok=True)
    
    # Use cached client if available
    try:
        # Initialize Supabase client if needed
        if supabase is None:
            from supabase_config import get_supabase_client
            supabase = get_supabase_client()
            
        # Check if bucket exists first
        try:
            # Import StorageException here to ensure it's available
            try:
                from storage3.utils import StorageException
            except ImportError:
                logger.warning("Could not import StorageException from storage3.utils")
            
            # Try to get bucket details - this will fail if bucket doesn't exist
            try:
                bucket_details = supabase.storage.get_bucket(BUCKET_NAME)
                logging.info(f"Storage bucket '{BUCKET_NAME}' already exists")
                
                # Ensure bucket is public by updating settings
                try:
                    supabase.storage.update_bucket(
                        BUCKET_NAME, 
                        options={
                            'public': True,
                            'file_size_limit': 52428800  # 50MB
                        }
                    )
                    logging.info(f"Updated bucket '{BUCKET_NAME}' settings to ensure public access")
                except Exception as e:
                    logging.warning(f"Could not update bucket settings: {str(e)}")
                    
            except Exception as e:
                # Bucket doesn't exist, create it
                logging.info(f"Storage bucket '{BUCKET_NAME}' not found, creating...")
                
                try:
                    bucket_options = {
                        'public': True,
                        'file_size_limit': 52428800  # 50MB
                    }
                    supabase.storage.create_bucket(BUCKET_NAME, options=bucket_options)
                    logging.info(f"Created storage bucket '{BUCKET_NAME}' with public access")
                except Exception as create_err:
                    logging.error(f"Failed to create storage bucket: {str(create_err)}")
                    raise
                    
        except Exception as bucket_err:
            logging.error(f"Error handling storage bucket: {str(bucket_err)}")
            
        # Run diagnostics to verify setup
        diagnostics = run_storage_diagnostics()
        logging.debug(f"Storage diagnostics: {diagnostics}")
            
    except Exception as e:
        logging.warning(f"Supabase storage initialization failed: {str(e)}")
        logging.info("Falling back to local storage only.")

def save_report(html_content, filename, upload_to_storage=True):
    """Save the report to local file and optionally to Supabase storage."""
    try:
        # Ensure HTML reports have required JavaScript functions to fix iframe display issues
        if filename.endswith('.html'):
            html_content = ensure_report_has_js_functions(html_content)
            
        # Always save locally first
        local_path = os.path.join('results', filename)
        with open(local_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logging.info(f"Report saved locally: {local_path}")
        
        # Upload to Supabase if requested
        if upload_to_storage:
            try:
                # Initialize storage if needed
                if supabase is None:
                    init_storage()
                
                # Check if the file already exists
                try:
                    exists = check_file_exists(filename)
                    if exists:
                        logging.info(f"File {filename} already exists in storage, replacing...")
                        # Delete existing file before upload
                        try:
                            supabase.storage.from_(BUCKET_NAME).remove([filename])
                            logging.info(f"Deleted existing file {filename} before re-upload")
                        except Exception as delete_err:
                            logging.warning(f"Error removing existing file: {str(delete_err)}")
                except Exception as check_err:
                    logging.warning(f"Could not check if file exists: {str(check_err)}")
                
                # Check bucket exists - recreate if needed
                try:
                    bucket_details = supabase.storage.get_bucket(BUCKET_NAME)
                except Exception as bucket_err:
                    logging.warning(f"Bucket error during save: {str(bucket_err)}")
                    logging.info("Attempting to recreate bucket...")
                    init_storage()
                
                # Upload file - Using the local file path instead of BytesIO due to compatibility issues
                with open(local_path, 'rb') as file_data:
                    supabase.storage.from_(BUCKET_NAME).upload(
                        file=file_data.read(),  # Read the file content as bytes
                        path=filename,
                        file_options={"content-type": "text/html"}
                    )
                
                # Get URL for the file
                file_url = get_file_url(filename)
                logging.info(f"Report uploaded to Supabase: {file_url}")
                
                return {
                    'local_path': local_path,
                    'storage_url': file_url
                }
                
            except Exception as e:
                logging.error(f"Failed to upload report to Supabase: {str(e)}")
                return {
                    'local_path': local_path,
                    'error': str(e)
                }
        
        return {'local_path': local_path}
    
    except Exception as e:
        logging.error(f"Failed to save report: {str(e)}")
        raise

def ensure_report_has_js_functions(html_content):
    """Ensure HTML reports have the proper Content Security Policy for direct display."""
    # Check if the report already has a permissive CSP meta tag
    if '<meta http-equiv="Content-Security-Policy"' in html_content and ('*' in html_content or "'unsafe-inline'" in html_content):
        return html_content
        
    # Add or replace CSP meta tag to allow all sources and inline scripts/styles
    csp_meta = '<meta http-equiv="Content-Security-Policy" content="default-src *; style-src * \'unsafe-inline\'; script-src * \'unsafe-inline\'; img-src * data:; connect-src *;">'
    
    # Replace existing CSP tag if present
    if '<meta http-equiv="Content-Security-Policy"' in html_content:
        import re
        html_content = re.sub(
            r'<meta http-equiv="Content-Security-Policy"[^>]*>',
            csp_meta,
            html_content
        )
    # Add CSP if not present
    elif '<head>' in html_content:
        html_content = html_content.replace(
            '<head>',
            f'<head>\n    {csp_meta}'
        )
    
    # Add JavaScript functions for collapsible sections and tabs
    js_functions = """
    <script>
        // Check if toggleCollapse is already defined
        if (typeof toggleCollapse !== 'function') {
            function toggleCollapse(id) {
                var content = document.getElementById(id);
                if (content.style.display === "block") {
                    content.style.display = "none";
                } else {
                    content.style.display = "block";
                }
            }
        }
        
        // Check if openTab is already defined
        if (typeof openTab !== 'function') {
            function openTab(evt, tabName) {
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tabcontent");
                for (i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].style.display = "none";
                }
                tablinks = document.getElementsByClassName("tablinks");
                for (i = 0; i < tablinks.length; i++) {
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }
                document.getElementById(tabName).style.display = "block";
                evt.currentTarget.className += " active";
            }
            
            // Initialize tabs if they exist
            document.addEventListener('DOMContentLoaded', function() {
                var defaultOpen = document.getElementById("defaultOpen");
                if (defaultOpen) {
                    defaultOpen.click();
                }
            });
        }
    </script>
    """
    
    # Add the JavaScript functions before the closing </body> tag
    if '</body>' in html_content:
        html_content = html_content.replace('</body>', f'{js_functions}\n</body>')
    else:
        # If no </body> tag, add before the closing </html> tag
        html_content = html_content.replace('</html>', f'{js_functions}\n</html>')
    
    return html_content

def get_file_url(filename):
    """Get public or signed URL for a file in storage."""
    if supabase is None:
        init_storage()
    
    # First try to get public URL
    try:
        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(filename)
        
        # Check if URL format is correct (handle different Supabase versions)
        if "/storage/v1/" in public_url and "/object/public" not in public_url:
            parts = public_url.split("/storage/v1/")
            if len(parts) == 2:
                base = parts[0]
                rest = parts[1]
                fixed_url = f"{base}/storage/v1/object/public/{rest}"
                logging.info(f"Fixed URL format from {public_url} to {fixed_url}")
                
                # Validate the URL works
                try:
                    import requests
                    response = requests.head(fixed_url, timeout=5)
                    if response.status_code < 400:
                        return fixed_url
                    else:
                        logging.warning(f"Fixed URL returned status code {response.status_code}")
                except Exception as e:
                    logging.warning(f"Could not validate fixed URL: {str(e)}")
                
                # Return the fixed URL even if we couldn't validate it
                return fixed_url
        
        return public_url
        
    except Exception as e:
        logging.warning(f"Failed to get public URL, trying signed URL: {str(e)}")
        
        # Fallback to signed URL
        try:
            signed_url = supabase.storage.from_(BUCKET_NAME).create_signed_url(
                path=filename,
                expires_in=3600  # 1 hour
            )
            return signed_url
        except Exception as e2:
            logging.error(f"Failed to get signed URL: {str(e2)}")
            
            # Return local file path as fallback
            local_path = f"/local_reports/{filename}"
            logging.info(f"Falling back to local file URL: {local_path}")
            return local_path

def get_report_from_storage(filename):
    """Get a report from Supabase storage. Assumes JSON content for now."""
    try:
        response = supabase.storage.from_(BUCKET_NAME).download(filename)
        # The download response is bytes, decode it to string then parse JSON
        data_str = response.decode('utf-8')
        data = json.loads(data_str)
        
        logger.info(f"Retrieved report from storage: {filename}")
        return data
    except Exception as e:
        logger.error(f"Error retrieving report from storage ({filename}): {e}")
        # Fallback to local JSON file still useful here
        try:
            local_path = os.path.join('results', filename)
            if os.path.exists(local_path):
                # Assuming local files are also JSON, matching the original intent
                with open(local_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"Retrieved report from local storage: {local_path}")
                return data
        except Exception as local_error:
            logger.error(f"Error retrieving report locally ({filename}): {local_error}")
        return None

def delete_report_from_storage(filename):
    """Delete a report from Supabase storage"""
    try:
        # Delete from Supabase
        supabase.storage.from_(BUCKET_NAME).remove([filename])
        
        logger.info(f"Deleted report from storage: {filename}")
        return True
    except Exception as e:
        logger.error(f"Error deleting report from storage: {e}")
        
        # Try to delete locally as fallback
        try:
            local_path = os.path.join('results', filename)
            if os.path.exists(local_path):
                os.remove(local_path)
                logger.info(f"Deleted report from local storage: {local_path}")
                return True
        except Exception as local_error:
            logger.error(f"Error deleting report locally: {local_error}")
            
        return False

def get_signed_url(filename, expires_in=3600):
    """Get a signed URL for a report"""
    try:
        # Get signed URL
        response = supabase.storage.from_(BUCKET_NAME).create_signed_url(
            path=filename,
            expires_in=expires_in
        )
        
        logger.info(f"Created signed URL for report: {filename}")
        return response['signedURL']
    except Exception as e:
        logger.error(f"Error creating signed URL: {e}")
        
        # Return local file path as fallback
        try:
            local_path = os.path.join('results', filename)
            if os.path.exists(local_path):
                return f"file://{os.path.abspath(local_path)}"
        except Exception:
            pass
            
        return None 

def check_file_exists(filename):
    """Check if a file exists in the storage bucket."""
    if supabase is None:
        init_storage()
    
    try:
        files = supabase.storage.from_(BUCKET_NAME).list()
        
        # Handle different file structures
        for file_obj in files:
            try:
                # Try dictionary access
                if hasattr(file_obj, 'get'):
                    name = file_obj.get('name')
                # Try attribute access
                elif hasattr(file_obj, 'name'):
                    name = file_obj.name
                # Last resort - string representation
                else:
                    name = str(file_obj)
                    
                if name == filename:
                    return True
            except Exception as e:
                logging.warning(f"Error checking file object: {str(e)}")
                continue
                
        return False
    except Exception as e:
        logging.error(f"Error checking if file exists: {str(e)}")
        # Assume file doesn't exist if there's an error
        return False

# Initialize storage when module is loaded
if __name__ != "__main__":  # Only run when imported as a module
    try:
        logger.info("Auto-initializing storage system")
        init_storage()
    except Exception as e:
        logger.error(f"Failed to auto-initialize storage: {e}")
        logger.error("Storage will be initialized when first used") 
