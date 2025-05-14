-- Supabase SQL Script to Update analysis_jobs Table

-- Ensure you have a backup before running any schema migration scripts.

-- Add 'total_steps' column if it doesn't exist
ALTER TABLE public.analysis_jobs
ADD COLUMN IF NOT EXISTS total_steps INTEGER DEFAULT 10;

COMMENT ON COLUMN public.analysis_jobs.total_steps IS 'Total number of steps anticipated for the analysis job.';

-- Add 'storage_url' column if it doesn't exist
ALTER TABLE public.analysis_jobs
ADD COLUMN IF NOT EXISTS storage_url TEXT;

COMMENT ON COLUMN public.analysis_jobs.storage_url IS 'URL to the stored analysis report (e.g., HTML report in cloud storage).';

-- Add 'raw_results' column if it doesn't exist (using JSONB for better performance and features)
ALTER TABLE public.analysis_jobs
ADD COLUMN IF NOT EXISTS raw_results JSONB;

COMMENT ON COLUMN public.analysis_jobs.raw_results IS 'Full JSON payload of the analysis results from the AI engine.';

-- Ensure 'end_time' column exists and is of type TIMESTAMPTZ (timestamp with time zone)
-- If it exists but has a different type, you might need to drop and recreate or cast, which is more complex.
-- This script just adds it if it's missing.
ALTER TABLE public.analysis_jobs
ADD COLUMN IF NOT EXISTS end_time TIMESTAMPTZ;

COMMENT ON COLUMN public.analysis_jobs.end_time IS 'Timestamp when the job ended (completed or failed).';

-- Ensure 'error' column exists and is of type TEXT
ALTER TABLE public.analysis_jobs
ADD COLUMN IF NOT EXISTS error TEXT;

COMMENT ON COLUMN public.analysis_jobs.error IS 'Error message if the job failed.';

-- Add 'guest_session_id' column if it doesn't exist (added for guest mode)
ALTER TABLE public.analysis_jobs
ADD COLUMN IF NOT EXISTS guest_session_id TEXT;

COMMENT ON COLUMN public.analysis_jobs.guest_session_id IS 'Session ID for guest users who initiated the job.';

-- Ensure user_id column is nullable (added for guest mode)
ALTER TABLE public.analysis_jobs
ALTER COLUMN user_id DROP NOT NULL; -- Run only if it's currently NOT NULL

-- Add 'keyword' column if it doesn't exist (based on recent error)
ALTER TABLE public.analysis_jobs
ADD COLUMN IF NOT EXISTS keyword TEXT;

COMMENT ON COLUMN public.analysis_jobs.keyword IS 'Target keyword for the SEO analysis.';

-- Optional: Update existing rows to have a default total_steps if they are NULL and it makes sense for your data
-- For example, if you want old jobs to also have a default of 10, uncomment and run:
-- UPDATE public.analysis_jobs
-- SET total_steps = 10
-- WHERE total_steps IS NULL;

-- Optional: Ensure other existing columns are correctly typed and nullable as expected by data_access.py
-- Example for 'progress' (assuming it should be INTEGER and can be NULL, though current code sets it):
-- ALTER TABLE public.analysis_jobs
-- ALTER COLUMN progress TYPE INTEGER,
-- ALTER COLUMN progress SET DEFAULT 0;

-- Example for 'steps_completed' (assuming it should be INTEGER and can be NULL):
-- ALTER TABLE public.analysis_jobs
-- ALTER COLUMN steps_completed TYPE INTEGER,
-- ALTER COLUMN steps_completed SET DEFAULT 0;


-- Note on existing data for 'total_steps' in create_analysis_job from data_access.py:
-- The Python code now defaults `total_steps` to 10 during job creation if not provided.
-- So new jobs will have this value.

-- Note on 'status' column (TEXT, e.g., 'pending', 'running', 'completed', 'failed')
-- Ensure it exists, e.g.:
-- ALTER TABLE public.analysis_jobs
-- ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'pending';

-- Note on 'pages' column (INTEGER)
-- Ensure it exists, e.g.:
-- ALTER TABLE public.analysis_jobs
-- ADD COLUMN IF NOT EXISTS pages INTEGER;


logger.info("Supabase schema script for analysis_jobs generated. Review and run in your Supabase SQL editor."); 