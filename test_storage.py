import os
import sys
import logging
import json
from datetime import datetime
from io import BytesIO

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('storage_test')

def test_supabase_client():
    """Test Supabase client initialization."""
    try:
        from supabase_config import get_supabase_client
        logger.info("Trying to initialize Supabase client...")
        
        client = get_supabase_client(use_service_role=True)
        logger.info("✅ Supabase client initialized successfully")
        return client
    except Exception as e:
        logger.error(f"❌ Failed to initialize Supabase client: {str(e)}")
        return None

def test_storage_module():
    """Test Storage module imports."""
    try:
        import storage3
        logger.info(f"✅ storage3 module found: {storage3.__version__}")
    except ImportError:
        logger.error("❌ storage3 module not found")
    
    try:
        from storage3.utils import StorageException
        logger.info("✅ StorageException imported from storage3.utils")
    except ImportError:
        logger.error("❌ Failed to import StorageException from storage3.utils")
    
    # Try importing our own storage module
    try:
        import storage
        logger.info("✅ Local storage module imported")
        
        # Check important attributes
        if hasattr(storage, 'init_storage'):
            logger.info("✅ init_storage function found")
        else:
            logger.error("❌ init_storage function not found in storage module")
            
        if hasattr(storage, 'save_report'):
            logger.info("✅ save_report function found")
        else:
            logger.error("❌ save_report function not found in storage module")
            
        # Check if old function still exists (should have been replaced)
        if hasattr(storage, 'save_report_to_storage'):
            logger.warning("⚠️ Old save_report_to_storage function still exists")
            
    except ImportError as e:
        logger.error(f"❌ Failed to import local storage module: {str(e)}")

def test_bucket_operations(client):
    """Test bucket operations."""
    if not client:
        logger.error("❌ Cannot test bucket operations: No Supabase client")
        return
    
    BUCKET_NAME = 'seo-reports'
    
    try:
        # List buckets
        logger.info("Listing buckets...")
        buckets = client.storage.list_buckets()
        
        bucket_names = []
        for bucket in buckets:
            try:
                if hasattr(bucket, 'name'):
                    name = bucket.name
                elif hasattr(bucket, 'get'):
                    name = bucket.get('name')
                else:
                    name = str(bucket)
                bucket_names.append(name)
            except Exception as e:
                logger.error(f"❌ Error getting bucket name: {str(e)}")
        
        logger.info(f"Found buckets: {bucket_names}")
        
        # Check if our bucket exists
        if BUCKET_NAME in bucket_names:
            logger.info(f"✅ Bucket '{BUCKET_NAME}' exists")
        else:
            logger.warning(f"⚠️ Bucket '{BUCKET_NAME}' not found, will try to create it")
            try:
                client.storage.create_bucket(BUCKET_NAME, {'public': True})
                logger.info(f"✅ Created bucket '{BUCKET_NAME}'")
            except Exception as e:
                logger.error(f"❌ Failed to create bucket: {str(e)}")
        
        # Try to update bucket settings
        try:
            logger.info(f"Updating bucket '{BUCKET_NAME}' settings...")
            client.storage.update_bucket(
                BUCKET_NAME, 
                options={
                    'public': True,
                    'file_size_limit': 52428800  # 50MB
                }
            )
            logger.info(f"✅ Updated bucket '{BUCKET_NAME}' settings")
        except Exception as e:
            logger.error(f"❌ Failed to update bucket settings: {str(e)}")
        
        # Try to upload a test file
        try:
            from datetime import datetime
            test_content = f"Test file created at {datetime.now().isoformat()}"
            test_filename = f"test_{int(datetime.now().timestamp())}.txt"
            
            # Create temp file for upload
            temp_path = os.path.join('tmp', test_filename)
            os.makedirs('tmp', exist_ok=True)
            
            with open(temp_path, 'w') as f:
                f.write(test_content)
            
            logger.info(f"Uploading test file '{test_filename}'...")
            
            with open(temp_path, 'rb') as f:
                client.storage.from_(BUCKET_NAME).upload(
                    file=f.read(),  # Read the file content as bytes
                    path=test_filename,
                    file_options={"content-type": "text/plain"}
                )
            logger.info(f"✅ Uploaded test file '{test_filename}'")
            
            # Try to get the URL
            try:
                logger.info(f"Getting public URL for '{test_filename}'...")
                public_url = client.storage.from_(BUCKET_NAME).get_public_url(test_filename)
                logger.info(f"✅ Got public URL: {public_url}")
                
                # Check if URL needs fixing
                if "/storage/v1/" in public_url and "/object/public" not in public_url:
                    parts = public_url.split("/storage/v1/")
                    if len(parts) == 2:
                        base = parts[0]
                        rest = parts[1]
                        fixed_url = f"{base}/storage/v1/object/public/{rest}"
                        logger.info(f"URL may need fixing. Original: {public_url}")
                        logger.info(f"Fixed URL: {fixed_url}")
                        
                        # Test both URLs
                        import requests
                        
                        try:
                            logger.info(f"Testing original URL with HEAD request...")
                            orig_response = requests.head(public_url, timeout=5)
                            logger.info(f"✅ Original URL status: {orig_response.status_code}")
                        except Exception as e:
                            logger.error(f"❌ Error testing original URL: {str(e)}")
                        
                        try:
                            logger.info(f"Testing fixed URL with HEAD request...")
                            fixed_response = requests.head(fixed_url, timeout=5)
                            logger.info(f"✅ Fixed URL status: {fixed_response.status_code}")
                        except Exception as e:
                            logger.error(f"❌ Error testing fixed URL: {str(e)}")
            except Exception as e:
                logger.error(f"❌ Failed to get public URL: {str(e)}")
            
            # Try to clean up the test file
            try:
                logger.info(f"Removing test file '{test_filename}'...")
                client.storage.from_(BUCKET_NAME).remove([test_filename])
                logger.info(f"✅ Removed test file '{test_filename}'")
            except Exception as e:
                logger.error(f"❌ Failed to remove test file: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Failed to upload test file: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Error during bucket operations: {str(e)}")

def test_our_storage_functions():
    """Test our storage.py functions."""
    try:
        from storage import init_storage, save_report, get_file_url, BUCKET_NAME
        
        logger.info("Testing init_storage...")
        init_storage()
        logger.info("✅ init_storage completed without errors")
        
        # Try saving a test report
        test_content = f"<html><body>Test report created at {datetime.now().isoformat()}</body></html>"
        test_filename = f"test_{int(datetime.now().timestamp())}.html"
        
        try:
            logger.info(f"Testing save_report with '{test_filename}'...")
            result = save_report(test_content, test_filename, upload_to_storage=True)
            logger.info(f"✅ save_report result: {result}")
            
            # Try getting the URL
            if 'storage_url' in result:
                try:
                    url = result['storage_url']
                    logger.info(f"Testing file URL: {url}")
                    
                    # Test accessing the URL
                    import requests
                    try:
                        response = requests.head(url, timeout=5)
                        logger.info(f"✅ URL HEAD status: {response.status_code}")
                    except Exception as e:
                        logger.error(f"❌ Error testing URL: {str(e)}")
                except Exception as e:
                    logger.error(f"❌ Error testing storage URL: {str(e)}")
        except Exception as e:
            logger.error(f"❌ Failed to save test report: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Error testing storage functions: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting Supabase storage diagnostics")
    
    # Test module imports
    test_storage_module()
    
    # Test Supabase client
    client = test_supabase_client()
    
    # Test bucket operations
    if client:
        test_bucket_operations(client)
    
    # Test our storage functions
    test_our_storage_functions()
    
    logger.info("Finished storage diagnostics") 