# AI modified: 2026-04-19 21:12:31 c1b6021e
"""Centralized filter table specification registry for metadata indexes."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FilterTableSpec:
    """Specification for a metadata filter table."""

    field_name: str
    table_name: str
    value_column: str
    supports_exact: bool
    supports_range: bool


def get_default_filter_specs() -> dict[str, FilterTableSpec]:
    """
    Return the default filter table specifications as a dict[str, FilterTableSpec].

    This dict is keyed by field_name (e.g., "sub_domain", "crawl_date").

    To add a new filter field:
    1. Define a FilterTableSpec with field_name, table_name, value_column, and
        predicate type support.
    2. Add it to the specs list below.

    To override at runtime, pass a dict[str, FilterTableSpec] to the index constructor's
        filter_table_specs parameter.
    """
    specs = [
        FilterTableSpec(
            field_name="sub_domain",
            table_name="metadata_sub_domain",
            value_column="sub_domain",
            supports_exact=True,
            supports_range=False,
        ),
        FilterTableSpec(
            field_name="crawl_date",
            table_name="metadata_crawl_date",
            value_column="crawl_date",
            supports_exact=False,
            supports_range=True,
        ),
    ]
    return {spec.field_name: spec for spec in specs}
