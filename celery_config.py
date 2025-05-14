from celery import Celery
import os

def get_postgresql_url_from_supabase():
    """Try to get PostgreSQL URL from Supabase config or return None if not available"""
    try:
        from supabase_config import get_postgresql_url
        return get_postgresql_url()
    except (ImportError, ValueError):
        return None

def make_celery(app=None):
    # Try to get PostgreSQL URL from Supabase
    postgres_url = get_postgresql_url_from_supabase()
    
    # Configure Celery with PostgreSQL if available, otherwise use filesystem
    if postgres_url:
        broker = f'db+{postgres_url}'
        backend = f'db+{postgres_url}'
    else:
        broker = 'filesystem://'
        backend = 'file://./celery_results'
    
    celery = Celery(
        'seo_analyzer',
        broker=broker,
        backend=backend,
        include=['tasks']
    )
    
    # Set default configurations
    celery.conf.update(
        result_expires=3600,  # Results expire after 1 hour
        worker_max_tasks_per_child=1000,
        worker_prefetch_multiplier=1,
        task_track_started=True,
        task_time_limit=3600,  # 1 hour task time limit
        task_soft_time_limit=3000,  # 50 minutes soft time limit
    )
    
    # Add broker transport options if using filesystem
    if broker == 'filesystem://':
        celery.conf.update(
            broker_transport_options={
                'data_folder_in': './celery_broker/in',
                'data_folder_out': './celery_broker/out',
                'data_folder_processed': './celery_broker/processed'
            }
        )
    
    if app:
        # If we're using Flask, integrate with the app context
        class FlaskTask(celery.Task):
            def __call__(self, *args, **kwargs):
                with app.app_context():
                    return self.run(*args, **kwargs)
        
        celery.Task = FlaskTask
    
    return celery 