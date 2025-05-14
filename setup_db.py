import os
import sys
from datetime import datetime, timezone
import logging
import json
import requests
from supabase_config import get_supabase_client, SUPABASE_URL, SUPABASE_KEY

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('setup_db')

# Define table schemas - these can be executed directly in the Supabase SQL editor
# Note: We can't create tables with the client API, we would need direct SQL access

def create_default_user():
    """Create a default user in the database"""
    supabase = get_supabase_client()
    default_user_id = 'd7b6db0c-44ba-4b2a-b958-25f2e84c2b14'
    
    # Check if the analysis_jobs table exists
    try:
        response = supabase.table('analysis_jobs').select('count', count='exact').limit(1).execute()
        logger.info(f"analysis_jobs table exists, found {response.count} records")
    except Exception as e:
        logger.error(f"Error accessing analysis_jobs table: {str(e)}")
        logger.info("Please create the necessary tables in Supabase first.")
        logger.info("Here are the SQL commands to run in the Supabase SQL Editor:")
        print_sql_commands()
        return None
    
    # Try to insert the analysis job directly since we're getting an error with the users table
    # This is a workaround - the proper solution would be to create all required tables
    try:
        # Try a direct analysis job insertion with the UUID
        url = f"{SUPABASE_URL}/rest/v1/analysis_jobs"
        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        test_job = {
            'user_id': default_user_id,
            'filename': 'test_setup_job.json',
            'url': 'https://example.com',
            'pages': 5,
            'status': 'test',
            'progress': 0,
            'current_step': 'setup',
            'steps_completed': 0,
            'total_steps': 5,
            'start_time': datetime.now(timezone.utc).isoformat()
        }
        
        # Make the direct API call
        response = requests.post(url, headers=headers, data=json.dumps(test_job))
        
        if response.status_code in (200, 201):
            logger.info(f"Successfully created test job with user ID: {default_user_id}")
            return {"user_id": default_user_id, "status": "created_test_job"}
        else:
            logger.error(f"Failed to create test job: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Error creating test job: {str(e)}")
        return None

def print_sql_commands():
    """Print the SQL commands to create the necessary tables"""
    sql_commands = """
-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL UNIQUE,
    email TEXT UNIQUE,
    username TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_admin BOOLEAN DEFAULT FALSE
);

-- Create analysis_jobs table
CREATE TABLE IF NOT EXISTS analysis_jobs (
    id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL,
    filename TEXT UNIQUE NOT NULL,
    url TEXT NOT NULL,
    pages INTEGER DEFAULT 5,
    status TEXT DEFAULT 'pending',
    progress INTEGER DEFAULT 0,
    current_step TEXT DEFAULT 'initializing',
    steps_completed INTEGER DEFAULT 0,
    total_steps INTEGER DEFAULT 5,
    result_path TEXT,
    storage_url TEXT,
    start_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    end_time TIMESTAMP WITH TIME ZONE,
    estimated_time_remaining INTEGER,
    task_id TEXT,
    error TEXT
);

-- Create RLS policies
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE analysis_jobs ENABLE ROW LEVEL SECURITY;

-- Allow users to read their own data
CREATE POLICY users_read_own ON users
    FOR SELECT USING (auth.uid() = user_id);

-- Allow users to read their own analysis jobs
CREATE POLICY jobs_read_own ON analysis_jobs
    FOR SELECT USING (auth.uid() = user_id);
    
-- For testing, you can also add a policy to allow unauthenticated access:
CREATE POLICY jobs_read_all ON analysis_jobs
    FOR SELECT USING (true);
"""
    logger.info(sql_commands)

if __name__ == '__main__':
    logger.info("Setting up database...")
    
    # Attempt to create default user or test job
    result = create_default_user()
    if result:
        logger.info(f"Setup complete: {result}")
    else:
        logger.error("Setup failed. Please create the tables manually.")
        sys.exit(1)
    
    logger.info("Setup process completed.") 