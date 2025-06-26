import os
import sys
# DON'T CHANGE THIS !!!
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS
from flask_migrate import Migrate
from src.models.portfolio import db
from src.config import get_config
from src.routes.user import user_bp

def create_app(config_name=None):
    """Application factory pattern for creating Flask app"""
    app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), 'static'))
    
    # Load configuration
    config_obj = get_config(config_name)
    app.config.from_object(config_obj)
    
    # Initialize extensions
    db.init_app(app)
    migrate = Migrate(app, db)
    
    # Enable CORS for all routes
    CORS(app, origins=app.config['CORS_ORIGINS'])
    
    # Register blueprints
    app.register_blueprint(user_bp, url_prefix='/api/v1/users')
    
    # Import and register additional blueprints
    from src.routes.portfolio import portfolio_bp
    from src.routes.analytics import analytics_bp
    from src.routes.mcp import mcp_bp
    from src.routes.reports import reports_bp
    from src.routes.agents import agents_bp
    
    app.register_blueprint(portfolio_bp, url_prefix='/api/v1')
    app.register_blueprint(analytics_bp, url_prefix='/api/v1')
    app.register_blueprint(mcp_bp, url_prefix='/api/v1')
    app.register_blueprint(reports_bp, url_prefix='/api/v1')
    app.register_blueprint(agents_bp, url_prefix='/api/v1')
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        return jsonify({
            'status': 'healthy',
            'version': '1.0.0',
            'environment': app.config.get('ENV', 'development'),
            'database': 'connected' if db.engine else 'disconnected'
        })
    
    # API info endpoint
    @app.route('/api/v1/info')
    def api_info():
        return jsonify({
            'name': 'Digi-Cadence Portfolio Management Platform',
            'version': '1.0.0',
            'description': 'Enterprise-grade multi-brand, multi-project portfolio management platform',
            'features': [
                'Multi-tenant architecture',
                'Portfolio optimization',
                'Advanced analytics',
                'Multi-dimensional reporting',
                'MCP server integration',
                'Multi-agent system'
            ],
            'endpoints': {
                'portfolio': '/api/v1/portfolio',
                'analytics': '/api/v1/analytics',
                'reports': '/api/v1/reports',
                'mcp': '/api/v1/mcp',
                'users': '/api/v1/users'
            }
        })
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Resource not found'}), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        return jsonify({'error': 'Internal server error'}), 500
    
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({'error': 'Bad request'}), 400
    
    @app.errorhandler(401)
    def unauthorized(error):
        return jsonify({'error': 'Unauthorized'}), 401
    
    @app.errorhandler(403)
    def forbidden(error):
        return jsonify({'error': 'Forbidden'}), 403
    
    # Frontend serving routes
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        static_folder_path = app.static_folder
        if static_folder_path is None:
            return jsonify({'error': 'Static folder not configured'}), 404

        if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
            return send_from_directory(static_folder_path, path)
        else:
            index_path = os.path.join(static_folder_path, 'index.html')
            if os.path.exists(index_path):
                return send_from_directory(static_folder_path, 'index.html')
            else:
                return jsonify({
                    'message': 'Digi-Cadence Portfolio Management Platform API',
                    'version': '1.0.0',
                    'documentation': '/api/v1/info'
                })
    
    return app

# Create the application instance
app = create_app()

if __name__ == '__main__':
    with app.app_context():
        # Create all database tables
        db.create_all()
        print("Database tables created successfully!")
        print(f"Starting Digi-Cadence Portfolio Management Platform...")
        print(f"Environment: {app.config.get('ENV', 'development')}")
        print(f"Database: {app.config['SQLALCHEMY_DATABASE_URI']}")
    
    app.run(host='0.0.0.0', port=5000, debug=app.config.get('DEBUG', False))
