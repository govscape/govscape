from flask import Flask, jsonify, request

app = Flask(__name__)


@app.route("/api/search/", methods=["POST"])
def search():
    data = request.get_json() or {}
    # Echo back weights and provide empty results so frontend can exercise UI
    return jsonify(
        {
            "results": [],
            "pagination": {
                "page": data.get("page", 1),
                "page_size": 20,
                "has_next_page": False,
                "total_count": 0,
                "total_pages": 0,
            },
            "received": data,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
