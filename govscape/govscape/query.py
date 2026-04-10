# This file defines a user query and the response that will be returned to the user.
# These two together roughly define the API that we will use to communicate with the
# front-end.

from abc import ABC, abstractmethod


class Predicate(ABC):
    @abstractmethod
    def __str__(self):
        pass


class RangePredicate(Predicate):
    def __init__(
        self,
        field_name: str,
        min_val: str | float | None = None,
        max_val: str | float | None = None,
    ):
        self.field_name = field_name
        self.min_val = min_val
        self.max_val = max_val

    def __str__(self):
        return f"{self.field_name} between {self.min_val} and {self.max_val}"


class EqualityPredicate(Predicate):
    def __init__(self, field_name: str, val: str | float):
        self.field_name = field_name
        self.value = val

    def __str__(self):
        return f"{self.field_name} = {self.value}"


class Query:
    def __init__(
        self,
        q_text: str,
        search_type: str,
        predicates: list[Predicate] | None = None,
        page: int = 1,
    ):
        self.q_text = q_text
        self.search_type = search_type
        self.predicates = predicates if predicates is not None else []
        self.page = page


class Response:
    def __init__(self, results: list, pagination: dict):
        self.results = results
        self.pagination = pagination

    def to_dict(self):
        return {
            "results": self.results,
            "pagination": self.pagination,
        }
