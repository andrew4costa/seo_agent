#!/usr/bin/env python
"""
SEO Agent Launcher
This script helps start all the components needed for the SEO Agent system
"""

import os
import sys
import argparse
import subprocess
import time
import signal
import atexit

def is_redis_running():
    """Check if Redis server is running"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        return True
    except:
        return False

def start_redis():
    """Start Redis server if not already running"""
    if is_redis_running():
        print("✅ Redis server is already running")
        return None
    
    print("🚀 Starting Redis server...")
    try:
        # Try to start Redis in the background
        redis_process = subprocess.Popen(['redis-server'], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE,
                                    universal_newlines=True)
        time.sleep(2)  # Give it a moment to start
        
        if redis_process.poll() is None:  # Still running
            print("✅ Redis server started")
            return redis_process
        else:
            print("❌ Failed to start Redis server")
            return None
    except FileNotFoundError:
        print("❌ Redis server not found. Please install Redis.")
        return None

def start_celery_worker():
    """Start Celery worker"""
    print("🚀 Starting Celery worker...")
    try:
        celery_process = subprocess.Popen(['celery', '-A', 'tasks.celery', 'worker', '--loglevel=info'],
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE,
                                    universal_newlines=True)
        time.sleep(3)  # Give it a moment to start
        
        if celery_process.poll() is None:  # Still running
            print("✅ Celery worker started")
            return celery_process
        else:
            print("❌ Failed to start Celery worker")
            return None
    except FileNotFoundError:
        print("❌ Celery not found. Check if it's installed correctly.")
        return None

def start_flask_app(host, port, debug=True):
    """Start Flask application"""
    print(f"🚀 Starting Flask application on {host}:{port}...")
    try:
        app_args = ['python', 'app.py', '--host', host, '--port', str(port)]
        flask_process = subprocess.Popen(app_args,
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE,
                                    universal_newlines=True)
        time.sleep(2)  # Give it a moment to start
        
        if flask_process.poll() is None:  # Still running
            print(f"✅ Flask application started at http://{host}:{port}")
            return flask_process
        else:
            print("❌ Failed to start Flask application")
            return None
    except Exception as e:
        print(f"❌ Error starting Flask app: {e}")
        return None

def cleanup_processes(processes):
    """Clean up all started processes"""
    print("\n🧹 Cleaning up processes...")
    for process in processes:
        if process and process.poll() is None:  # Still running
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ Process {process.pid} terminated")
            except:
                process.kill()
                print(f"✅ Process {process.pid} killed")

def main():
    parser = argparse.ArgumentParser(description="Start the SEO Agent system components")
    parser.add_argument("--host", default="127.0.0.1", help="Host to run the Flask app on")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the Flask app on")
    parser.add_argument("--no-redis", action="store_true", help="Don't start Redis (use if already running)")
    parser.add_argument("--no-celery", action="store_true", help="Don't start Celery worker (use if already running)")
    parser.add_argument("--prod", action="store_true", help="Run in production mode with Gunicorn")
    
    args = parser.parse_args()
    processes = []
    
    print("🚀 Starting SEO Agent system...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️ Warning: .env file not found. Make sure environment variables are set.")
    
    # Start Redis if needed
    if not args.no_redis:
        redis_process = start_redis()
        if redis_process:
            processes.append(redis_process)
    
    # Start Celery worker if needed
    if not args.no_celery:
        celery_process = start_celery_worker()
        if celery_process:
            processes.append(celery_process)
    
    # Start Flask app
    if args.prod:
        try:
            # Use Gunicorn for production
            workers = os.cpu_count() * 2 + 1  # Recommended number of workers
            print(f"🚀 Starting production server with Gunicorn ({workers} workers)...")
            gunicorn_cmd = [
                'gunicorn', 
                '--workers', str(workers), 
                '--bind', f"{args.host}:{args.port}", 
                'app:app'
            ]
            flask_process = subprocess.Popen(gunicorn_cmd)
            processes.append(flask_process)
            print(f"✅ Production server started at http://{args.host}:{args.port}")
        except FileNotFoundError:
            print("❌ Gunicorn not found. Install it with 'pip install gunicorn' or run without --prod flag.")
            cleanup_processes(processes)
            sys.exit(1)
    else:
        # Development mode with Flask's built-in server
        flask_process = start_flask_app(args.host, args.port)
        if flask_process:
            processes.append(flask_process)
    
    # Register cleanup function to handle termination
    atexit.register(cleanup_processes, processes)
    
    # Print instructions
    print("\n✅ SEO Agent system is now running!")
    print(f"📊 Access the web interface at http://{args.host}:{args.port}")
    print("🛑 Press Ctrl+C to stop all components\n")
    
    try:
        # Keep the main process running
        while all(p.poll() is None for p in processes if p):
            time.sleep(1)
        
        # If we get here, one of the processes has stopped
        for p in processes:
            if p and p.poll() is not None:
                print(f"⚠️ Process exited with code {p.poll()}")
                # Print output for debugging
                stdout, stderr = p.communicate()
                if stdout:
                    print("Standard output:", stdout)
                if stderr:
                    print("Standard error:", stderr)
        
        # Clean up remaining processes
        cleanup_processes(processes)
        
    except KeyboardInterrupt:
        print("\n⚠️ Received keyboard interrupt, shutting down...")
        cleanup_processes(processes)
        print("✅ Shutdown complete")

if __name__ == "__main__":
    main() 