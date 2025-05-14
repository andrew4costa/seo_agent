#!/usr/bin/env python
from tasks import celery

if __name__ == '__main__':
    print("Starting Celery worker...")
    # This will make the worker use the filesystem backend instead of Redis
    # The worker will read from celery_broker/* directories
    celery.worker_main(['worker', '--loglevel=info']) 