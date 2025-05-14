import os
import json
import logging
from supabase import create_client, Client # Import Client for type hinting
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger('seo_agent')

# Supabase configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_API_KEY") # Public Anon Key
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY") # Service Role Key

# Cache clients to avoid recreating them frequently
_supabase_anon_client = None
_supabase_service_client = None

def get_supabase_url():
    """Returns the Supabase project URL."""
    if not SUPABASE_URL:
        raise ValueError("SUPABASE_URL environment variable is not set.")
    return SUPABASE_URL

def get_supabase_client(use_service_role: bool = False) -> Client:
    """Returns an initialized Supabase client.
    
    Args:
        use_service_role: If True, initializes client with the service role key.
                          Defaults to False (uses anon key).
    """
    global _supabase_anon_client, _supabase_service_client
    
    if not SUPABASE_URL:
        raise ValueError("SUPABASE_URL environment variable is not set.")

    if use_service_role:
        # Use cached service client if available
        if _supabase_service_client:
            return _supabase_service_client
            
        if not SUPABASE_SERVICE_KEY:
            raise ValueError("SUPABASE_SERVICE_KEY environment variable is not set, but use_service_role was True.")
        
        logger.info("Initializing Supabase client with service role key") 
        try:
            client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
            _supabase_service_client = client
            return client
        except Exception as e:
            logger.error(f"Error creating Supabase service role client: {e}")
            raise
    else:
        # Use cached anon client if available
        if _supabase_anon_client:
            return _supabase_anon_client
            
        if not SUPABASE_ANON_KEY:
            raise ValueError("SUPABASE_API_KEY (anon key) environment variable is not set.")
        
        logger.info("Initializing Supabase client with anon key")
        try:
            client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
            _supabase_anon_client = client
            return client
        except Exception as e:
            logger.error(f"Error creating Supabase anon client: {e}")
            raise

def test_supabase_connection():
    """Test Supabase connection and return diagnostic information."""
    results = {
        "supabase_url": SUPABASE_URL,
        "has_anon_key": bool(SUPABASE_ANON_KEY),
        "has_service_key": bool(SUPABASE_SERVICE_KEY),
        "anon_client_test": None,
        "service_client_test": None
    }
    
    # Test anon client
    try:
        client = get_supabase_client(use_service_role=False)
        # Simple query to test connection
        response = client.table('analysis_jobs').select('id').limit(1).execute()
        results["anon_client_test"] = {
            "connected": True,
            "data_sample": response.data[:1] if response.data else []
        }
    except Exception as e:
        results["anon_client_test"] = {
            "connected": False,
            "error": str(e)
        }
    
    # Test service role client
    try:
        client = get_supabase_client(use_service_role=True)
        # Simple query to test connection
        response = client.table('analysis_jobs').select('id').limit(1).execute()
        results["service_client_test"] = {
            "connected": True,
            "data_sample": response.data[:1] if response.data else []
        }
        
        # Test storage too
        try:
            buckets = client.storage.list_buckets()
            bucket_info = []
            for bucket in buckets:
                try:
                    # Extract bucket name (handling different client versions)
                    if hasattr(bucket, 'name'):
                        name = bucket.name
                    elif hasattr(bucket, 'get') and callable(bucket.get):
                        name = bucket.get('name', str(bucket))
                    else:
                        name = str(bucket)
                    
                    bucket_info.append(name)
                except Exception as bucket_err:
                    bucket_info.append(f"Error getting bucket info: {bucket_err}")
            
            results["storage_test"] = {
                "connected": True,
                "buckets": bucket_info
            }
        except Exception as storage_e:
            results["storage_test"] = {
                "connected": False,
                "error": str(storage_e)
            }
    except Exception as e:
        results["service_client_test"] = {
            "connected": False,
            "error": str(e)
        }
    
    return results

def get_postgresql_url():
    """
    Returns a PostgreSQL connection string for Celery and SQLAlchemy.
    Format: postgresql://postgres:[password]@db.[project-ref].supabase.co:5432/postgres
    
    Supabase provides direct database access which we can use for Celery.
    """
    db_password = os.environ.get("SUPABASE_DB_PASSWORD")
    project_id = os.environ.get("SUPABASE_PROJECT_ID")

    if not db_password or not project_id:
        raise ValueError("SUPABASE_DB_PASSWORD and SUPABASE_PROJECT_ID must be set for PostgreSQL URL.")

    db_host = os.environ.get("SUPABASE_DB_HOST", f"db.{project_id}.supabase.co")
    
    return f"postgresql://postgres:{db_password}@{db_host}:5432/postgres" 