# SEO Analyzer with Supabase Integration

A comprehensive SEO analysis tool that uses DataForSEO API and AI to provide actionable SEO recommendations.

## Features

- Technical SEO analysis
- On-page SEO evaluation
- Off-page SEO metrics
- User authentication and authorization
- Detailed reports with actionable recommendations
- API for integration with other tools

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- A Supabase account (https://supabase.com)
- DataForSEO API credentials
- Google Gemini API key (for AI recommendations)

### Environment Setup

1. Clone the repository
2. Create a virtual environment and activate it
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
4. Set up a Supabase project:
   - Go to https://supabase.com and create a new project
   - Get your Supabase URL, API Key, Project ID, and Database Password
   - Create a `.env` file with the following variables:
     ```
     # Supabase configuration
     SUPABASE_URL=https://YOUR_PROJECT_ID.supabase.co
     SUPABASE_API_KEY=YOUR_SUPABASE_API_KEY
     SUPABASE_DB_PASSWORD=YOUR_POSTGRES_DB_PASSWORD
     SUPABASE_PROJECT_ID=YOUR_PROJECT_ID
     
     # Flask configuration
     SECRET_KEY=your_flask_secret_key
     
     # DataForSEO API credentials
     DATAFORSEO_LOGIN=your_dataforseo_login
     DATAFORSEO_PASSWORD=your_dataforseo_password
     
     # Google Gemini API key 
     GEMINI_API_KEY=your_gemini_api_key
     ```

## Running the Application

1. Start the Flask application:
   ```
   python app.py --port 5001
   ```

2. Start the Celery worker (in a separate terminal):
   ```
   python start_worker.py
   ```

3. Access the application at http://localhost:5001

## Database Migrations

If you make changes to the database models, you'll need to run database migrations:

```
flask db init (if not already initialized)
flask db migrate -m "Your migration message"
flask db upgrade
```

## Architecture

- Flask: Web framework
- Supabase: Authentication, database, and storage
- Celery: Background task processing
- SQLAlchemy: ORM for database operations
- DataForSEO API: SEO data collection
- Google Gemini API: AI-powered recommendations

## Authentication with Supabase

This application uses Supabase for authentication, database, and file storage. The authentication flow is as follows:

1. Users sign up or log in through the Supabase Authentication service
2. After successful authentication, the user is redirected to the main application
3. User information is stored in Supabase's auth schema
4. All API requests include the user's JWT for authorization

### Custom Authorization Rules

User permissions are based on roles defined in your Supabase project:
- Regular users can only see and manage their own SEO reports
- Admin users can view all reports and manage system settings

To set up admin access, you can use Supabase's built-in Row Level Security (RLS) policies and custom claims.

## API Documentation

The application provides several API endpoints for integration with other tools:

- `/api/results/<filename>` - Get the results of an SEO analysis
- `/api/progress/<filename>` - Check the progress of an ongoing analysis
- `/api/user` - Get information about the current user

All API endpoints require authentication with a valid Supabase JWT.

## Scaling for Multiple Users

The SEO Agent is designed to handle multiple concurrent users through:

1. **Task Queue**: Analyses run as background Celery tasks, preventing web server blocking
2. **Database-backed Job Storage**: All analysis jobs are stored in a SQLite database (can be upgraded to PostgreSQL/MySQL for production)
3. **Rate Limiting**: Prevents individual users from overwhelming the system
4. **User Authentication**: Ensures users can only access their own reports
5. **Resource Cleanup**: Old analysis files are automatically archived to save disk space

## Production Deployment

For production deployment, consider:

1. Using a production WSGI server like Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 app:app
```

2. Running behind a reverse proxy like Nginx

3. Upgrading to a robust database like PostgreSQL:
```
DATABASE_URL=postgresql://username:password@localhost/dbname
```

4. Setting up a supervised Celery worker process

5. Upgrading to a Clerk paid plan for production usage

## Known Issues

### CSS Linting Errors in Templates

You may notice CSS linting errors in the template files related to inline style attributes with percentage values. These are benign linting issues that do not affect functionality:

- `at-rule or selector expected` 
- `property value expected`

These errors appear on lines with percentage-based width values like `style="width: {{ progress }}%"`. They are a limitation of the CSS linter and can be safely ignored.

## License

MIT License 