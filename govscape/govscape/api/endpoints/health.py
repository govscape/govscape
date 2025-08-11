from flask import current_app
from flask_restx import Namespace, Resource, fields

# Create namespace
ns = Namespace('health', description='Health check operations')

# Define model
health_model = ns.model('HealthResponse', {
    'status': fields.String(description='Server health status'),
    'embeddings_count': fields.Integer(description='Number of embeddings loaded')
})

@ns.route('/')
class HealthCheck(Resource):
    @ns.doc('get_health')
    @ns.response(200, 'Success', health_model)
    def get(self):
        """Get server health status"""
        server = current_app.server
        
        return {
            "status": "healthy", 
            "embeddings_count": server.text_index.total_entries()
        }
