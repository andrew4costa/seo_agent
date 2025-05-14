from flask_sqlalchemy import SQLAlchemy
import datetime
from datetime import timezone
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship

db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100), unique=True, nullable=False)  # Supabase/Clerk user ID
    email = db.Column(db.String(255), unique=True, nullable=False)
    username = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=lambda: datetime.datetime.now(timezone.utc))
    is_admin = db.Column(db.Boolean, default=False)
    
    # Relationships
    websites = relationship("Website", backref="user", cascade="all, delete-orphan")
    reports = relationship("Report", backref="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<User {self.email}>'

class Website(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, ForeignKey('user.id'), nullable=False)
    url = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=lambda: datetime.datetime.now(timezone.utc))
    
    # Relationships
    reports = relationship("Report", backref="website", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<Website {self.url}>'

class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, ForeignKey('user.id'), nullable=False)
    website_id = db.Column(db.Integer, ForeignKey('website.id'), nullable=False)
    title = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=lambda: datetime.datetime.now(timezone.utc))
    filename = db.Column(db.String(255), unique=True, nullable=False)
    storage_url = db.Column(db.String(512))  # URL to the file in Supabase storage
    status = db.Column(db.String(50), default='pending')
    is_paid = db.Column(db.Boolean, default=False)
    
    # Relationships
    tasks = relationship("Task", backref="report", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f'<Report {self.title}>'

class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(db.Integer, ForeignKey('report.id'), nullable=False)
    task_id = db.Column(db.String(100))  # Celery task ID
    type = db.Column(db.String(50), nullable=False)  # e.g., 'analysis', 'export', etc.
    status = db.Column(db.String(50), default='pending')
    created_at = db.Column(db.DateTime, default=lambda: datetime.datetime.now(timezone.utc))
    completed_at = db.Column(db.DateTime)
    
    def __repr__(self):
        return f'<Task {self.task_id}>'

class AnalysisJob(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100), nullable=False, index=True)  # Changed to string to store Clerk user IDs
    filename = db.Column(db.String(255), unique=True, nullable=False)
    url = db.Column(db.String(255), nullable=False)
    pages = db.Column(db.Integer, default=5)
    status = db.Column(db.String(50), default='pending')
    progress = db.Column(db.Integer, default=0)
    current_step = db.Column(db.String(100), default='initializing')
    steps_completed = db.Column(db.Integer, default=0)
    total_steps = db.Column(db.Integer, default=5)
    result_path = db.Column(db.String(255))
    storage_url = db.Column(db.String(512))  # URL to the file in Supabase storage
    start_time = db.Column(db.DateTime, default=lambda: datetime.datetime.now(timezone.utc))
    end_time = db.Column(db.DateTime)
    estimated_time_remaining = db.Column(db.Integer)
    task_id = db.Column(db.String(100))
    error = db.Column(db.Text)
    
    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'url': self.url,
            'status': self.status,
            'progress': self.progress,
            'current_step': self.current_step,
            'steps_completed': self.steps_completed,
            'total_steps': self.total_steps,
            'storage_url': self.storage_url,
            'start_time': self.start_time.timestamp() if self.start_time else None,
            'end_time': self.end_time.timestamp() if self.end_time else None,
            'estimated_time_remaining': self.estimated_time_remaining
        }
    
    def __repr__(self):
        return f'<AnalysisJob {self.filename}>' 