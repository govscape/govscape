# This file defines a user query and the response that will be returned to the user.
# These two together roughly define the API that we will use to communicate with the
# front-end.


class Query:
    def __init__(self, q_text, search_type, filters=None, page=1):
        self.q_text = q_text
        self.search_type = search_type
        self.filters = filters
        self.page = page


class Response:
    def __init__(self, results, pagination):
        self.results = results
        self.pagination = pagination

    def to_dict(self):
        return {
            "results": self.results,
            "pagination": self.pagination,
        }
