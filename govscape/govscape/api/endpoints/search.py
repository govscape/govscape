# AI modified: 2026-03-14 4a6b1b72
from flask import current_app, request
from flask_restx import Namespace, Resource, fields

from ...query import EqualityPredicate, Predicate, Query, RangePredicate

# Create namespace
ns = Namespace("search", description="Search operations")

# Define models
search_input = ns.model(
    "SearchInput",
    {
        "query": fields.String(required=True, description="Search query text"),
        "search_type": fields.String(required=True, description="Search query text"),
        "filters": fields.Raw(description="Filters to apply to the search"),
        "page": fields.Integer(description="Page number for pagination", default=1),
    },
)

search_crawl_instance = ns.model(
    "SearchCrawlInstance",
    {
        "crawl_url": fields.String(description="Source URL for this crawl"),
        "crawl_date": fields.String(description="Crawl date (YYYY-MM-DD)"),
        "sub_domain": fields.String(description="Subdomain for this crawl"),
    },
)

search_result = ns.model(
    "SearchResult",
    {
        "pdf": fields.String(description="PDF file path"),
        "page": fields.String(description="Page number"),
        "distance": fields.Float(description="Distance score"),
        "jpeg": fields.String(description="JPEG image path"),
        "crawl_date": fields.String(description="Most recent crawl date"),
        "crawl_url": fields.String(description="Most recent crawl URL"),
        "sub_domain": fields.String(description="Most recent subdomain"),
        "has_more_crawls": fields.Boolean(
            description="True when additional crawls exist beyond returned list"
        ),
        "crawl_instances": fields.List(
            fields.Nested(search_crawl_instance),
            description="Most recent crawl instances, newest first",
        ),
    },
)

pagination_model = ns.model(
    "Pagination",
    {
        "page": fields.Integer(description="Current page number"),
        "page_size": fields.Integer(description="Number of results per page"),
        "has_next_page": fields.Boolean(
            description="Indicates if there is a next page"
        ),
        "total_count": fields.Integer(description="Total number of PDFs"),
        "total_pages": fields.Integer(description="Total number of pages"),
    },
)

search_response = ns.model(
    "SearchResponse",
    {
        "results": fields.List(fields.Nested(search_result)),
        "pagination": fields.Nested(pagination_model),
    },
)


def convert_filters_to_predicates(filters: dict) -> list[Predicate]:
    predicates: list[Predicate] = []
    for ftype, val in filters.items():
        if not val:
            continue
        if ftype == "crawled_after":
            predicates.append(
                RangePredicate("crawl_date", min_val=val.replace("-", ""))
            )
        elif ftype == "crawled_before":
            predicates.append(
                RangePredicate("crawl_date", max_val=val.replace("-", ""))
            )
        elif ftype == "sub_domain":
            predicates.append(EqualityPredicate("sub_domain", val))
    return predicates


@ns.route("/")
class Search(Resource):
    @ns.doc("search_documents")
    @ns.expect(search_input, validate=True)
    @ns.response(200, "Success", search_response)
    @ns.response(400, "Invalid input")
    def post(self):
        """Search for documents matching the query"""
        data = request.get_json()
        if not data or "query" not in data:
            return {"status": "error", "message": "Missing 'query' parameter"}, 400

        if "search_type" not in data:
            return {
                "status": "error",
                "message": "Missing 'search_type' parameter",
            }, 400

        q_text = data.get("query")
        if not q_text.strip():
            return {"status": "error", "message": "Query cannot be empty"}, 400

        search_type = data.get("search_type")
        if not search_type.strip():
            return {"status": "error", "message": "search_type cannot be empty"}, 400

        predicates: list[Predicate] = []
        predicates = convert_filters_to_predicates(data.get("filters", {}))

        query = Query(
            q_text=q_text,
            search_type=search_type,
            predicates=predicates,
            page=data.get("page", 1),
        )

        server = current_app.server

        return server.search(query).to_dict()
