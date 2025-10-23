from functools import wraps
from flask import request, jsonify
import jwt

SECRET_KEY = 'ISTTSxUdayanaProject'

def token_required(allowed_roles=[]):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            token = None
            if 'Authorization' in request.headers:
                auth_header = request.headers['Authorization']
                if auth_header.startswith('Bearer '):
                    token = auth_header[7:]

            if not token:
                return jsonify({'error': 'Token is missing'}), 401

            try:
                decoded = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
                request.user = decoded
            except jwt.ExpiredSignatureError:
                return jsonify({'error': 'Token has expired'}), 401
            except jwt.InvalidTokenError:
                return jsonify({'error': 'Invalid token'}), 401

            if allowed_roles and decoded.get('role') not in allowed_roles:
                return jsonify({'error': 'Access denied: insufficient permissions'}), 403

            return f(*args, **kwargs)
        return wrapper
    return decorator