# This file defines a user query and the response that will be returned to the user.
# These two together roughly define the API that we will use to communicate with the
# front-end.


class Query:
    def __init__(self, q_text):
        self.q_text = q_text


class Response:
    def __init__(self, filename):
        self.filename = filename
