-- This is a reference schema for setting up your Supabase database
-- You should run these through the Supabase SQL editor

-- Create analysis_jobs table
CREATE TABLE IF NOT EXISTS public.analysis_jobs (
    id SERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id),
    filename VARCHAR(255) UNIQUE NOT NULL,
    url VARCHAR(255) NOT NULL,
    pages INTEGER DEFAULT 5,
    status VARCHAR(50) DEFAULT 'pending',
    progress INTEGER DEFAULT 0,
    current_step VARCHAR(100) DEFAULT 'initializing',
    steps_completed INTEGER DEFAULT 0,
    total_steps INTEGER DEFAULT 5,
    result_path VARCHAR(255),
    storage_url VARCHAR(512),
    start_time TIMESTAMP WITH TIME ZONE DEFAULT now(),
    end_time TIMESTAMP WITH TIME ZONE,
    estimated_time_remaining INTEGER,
    task_id VARCHAR(100),
    error TEXT
);

-- Create index for faster user-based queries
CREATE INDEX IF NOT EXISTS idx_analysis_jobs_user_id ON public.analysis_jobs(user_id);

-- Set up Row Level Security (RLS)
ALTER TABLE public.analysis_jobs ENABLE ROW LEVEL SECURITY;

-- Create policy to allow users to see only their own jobs
CREATE POLICY "Users can view their own analysis jobs"
    ON public.analysis_jobs
    FOR SELECT
    USING (auth.uid() = user_id);

-- Create policy to allow users to insert their own jobs
CREATE POLICY "Users can create their own analysis jobs"
    ON public.analysis_jobs
    FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- Create policy to allow users to update their own jobs
CREATE POLICY "Users can update their own analysis jobs"
    ON public.analysis_jobs
    FOR UPDATE
    USING (auth.uid() = user_id);

-- Create policy to allow users to delete their own jobs
CREATE POLICY "Users can delete their own analysis jobs"
    ON public.analysis_jobs
    FOR DELETE
    USING (auth.uid() = user_id);

-- Create admin role (if not already exists)
DO
$$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'admin') THEN
    CREATE ROLE admin;
  END IF;
END
$$;

-- Create policy for admins to manage all jobs
CREATE POLICY "Admins can manage all analysis jobs"
    ON public.analysis_jobs
    FOR ALL
    USING (auth.jwt() ->> 'role' = 'admin');

-- Function to get user role from JWT
CREATE OR REPLACE FUNCTION public.get_user_role()
RETURNS TEXT AS $$
BEGIN
  RETURN (auth.jwt() ->> 'role')::TEXT;
END;
$$ LANGUAGE plpgsql; 