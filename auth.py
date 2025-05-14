import functools
from flask import request, jsonify, current_app, g
# We will get the supabase client from current_app.extensions or pass it explicitly

def token_required(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(" ")[1]
            except IndexError:
                return jsonify({"message": "Bearer token malformed"}), 401

        if not token:
            return jsonify({"message": "Token is missing"}), 401

        try:
            # Assuming the supabase client is stored in app.extensions by Flask-Supabase or similar
            # Or that current_app has a direct reference like current_app.supabase
            # For now, let's assume it's accessible via current_app.supabase which we'll set up in app.py
            if not hasattr(current_app, 'supabase_client') or current_app.supabase_client is None:
                current_app.logger.error("Supabase client not initialized or not found on current_app.")
                return jsonify({"message": "Authentication service not available"}), 500

            user_response = current_app.supabase_client.auth.get_user(token)
            
            if user_response and user_response.user:
                g.user = user_response.user # Supabase user object
                # Ensure g.user.id is accessible. Supabase user object has an 'id' attribute.
            else:
                # This case might indicate a token that was once valid but user session is no longer,
                # or an issue with the Supabase client's ability to verify.
                current_app.logger.warning(f"Token validation failed or no user associated with token. Response: {user_response}")
                return jsonify({"message": "Token is invalid or session expired"}), 401
                
        except Exception as e:
            current_app.logger.error(f"Error during token validation: {str(e)}")
            # Check if the error message indicates an invalid token specifically
            if "invalid token" in str(e).lower() or "jwt expired" in str(e).lower():
                 return jsonify({"message": "Token is invalid or expired"}), 401
            return jsonify({"message": "Error processing token"}), 500

        return f(*args, **kwargs)
    return decorated_function

def get_current_user_id():
    if hasattr(g, 'user') and g.user:
        return g.user.id
    return None

def try_get_user_from_token():
    """Tries to authenticate a user if a token is present, but doesn't fail if not.
    Returns the Supabase user object if authentication is successful, otherwise None.
    Sets g.user if successful.
    """
    token = None
    if 'Authorization' in request.headers:
        auth_header = request.headers['Authorization']
        try:
            token = auth_header.split(" ")[1]
        except IndexError:
            current_app.logger.warning("Bearer token malformed during optional auth check.")
            return None # Malformed token, treat as no token

    if not token:
        return None # No token provided

    try:
        if not hasattr(current_app, 'supabase_client') or current_app.supabase_client is None:
            current_app.logger.error("Supabase client not available for optional auth check.")
            return None # Cannot authenticate

        user_response = current_app.supabase_client.auth.get_user(token)
        
        if user_response and user_response.user:
            g.user = user_response.user # Set g.user for this request if auth succeeds
            return user_response.user
        else:
            # Token was present but invalid or session expired
            current_app.logger.info(f"Optional auth: Token presented but validation failed or no user. Response: {user_response}")
            return None
            
    except Exception as e:
        # Log specific JWT errors differently from general exceptions if possible
        if "invalid token" in str(e).lower() or "jwt expired" in str(e).lower():
            current_app.logger.info(f"Optional auth: Invalid or expired token presented: {str(e)}")
        else:
            current_app.logger.error(f"Optional auth: Error during token validation: {str(e)}")
        return None # Error during validation, treat as no authenticated user 