from flask import current_app
from flask_restx import Namespace, Resource, fields

# Create namespace
ns = Namespace("pages", description="PDF pages operations")

pages_response = ns.model(
    "PagesResponse",
    {
        "images": fields.List(fields.String, description="List of image paths for pages"),
        "crawl_url": fields.String(description="Crawl URL for the PDF"),
        "crawl_date": fields.String(description="Crawl date for the PDF (YYYY-MM-DD)"),
        "sub_domain": fields.String(description="Subdomain for the PDF source"),
    },
)


@ns.route("/<pdf_id>")
@ns.param("pdf_id", "The PDF ID")
class Pages(Resource):
    @ns.doc("get_pdf_pages")
    @ns.response(200, "Success", pages_response)
    @ns.response(404, "PDF not found")
    @ns.response(400, "Invalid input")
    def get(self, pdf_id):
        """Get all page images for a PDF"""
        server = current_app.server
        return server.pdf_pages(pdf_id)
