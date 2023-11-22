"""Test the summary generation of the last request."""
import os

import toml

from nl2sql.requests_summary import RequestsSummaryBuilder


def test_build_summary() -> None:
    """Test the summary generation of the last request."""
    os.environ["OPENAI_API_KEY"] = toml.load(".streamlit/secrets.toml")["openai_api_key"]

    condensed_last_request = RequestsSummaryBuilder.build_summary(
        requests=("Show 10 customers", "Sorry I meant 2"),
    )

    assert condensed_last_request == "Show 2 customers."
