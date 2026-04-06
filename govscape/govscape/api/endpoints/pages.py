# AI modified: 2026-03-08 f62d40b8
# AI modified: 2026-03-14 4a6b1b72
from flask import current_app
from flask_restx import Namespace, Resource, fields

# Create namespace
ns = Namespace("pages", description="PDF pages operations")

crawl_instance_model = ns.model(
    "CrawlInstance",
    {
        "crawl_url": fields.String(description="Source URL for this crawl"),
        "crawl_date": fields.String(description="Crawl date (YYYY-MM-DD)"),
        "sub_domain": fields.String(description="Subdomain for this crawl"),
    },
)

pages_response = ns.model(
    "PagesResponse",
    {
        "images": fields.List(
            fields.String, description="List of image paths for pages"
        ),
        "crawl_url": fields.String(description="Most recent crawl URL"),
        "crawl_date": fields.String(description="Most recent crawl date"),
        "sub_domain": fields.String(description="Most recent subdomain"),
        "has_more_crawls": fields.Boolean(
            description="True when additional crawls exist beyond returned list"
        ),
        "crawl_instances": fields.List(
            fields.Nested(crawl_instance_model),
            description="Most recent crawl instances, newest first",
        ),
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
