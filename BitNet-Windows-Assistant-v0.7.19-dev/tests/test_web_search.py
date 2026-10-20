import unittest
from bitnet_windows.web_search import _DuckParser, WebResult, format_web_context, format_web_references

class WebSearchParserTests(unittest.TestCase):
    def test_parses_result_and_rejects_non_http_links(self):
        html = """<a class="result__a" href="https://example.org/page">Example page</a><a class="result__a" href="javascript:alert(1)">Bad</a>"""
        parser = _DuckParser(); parser.feed(html)
        self.assertEqual(len(parser.results), 1)
        self.assertEqual(parser.results[0]["url"], "https://example.org/page")
        self.assertEqual(parser.results[0]["title"], "Example page")
    def test_resolves_duckduckgo_redirect_to_http_destination(self):
        html = '<a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.org%2Farticle">Article</a>'
        parser = _DuckParser(); parser.feed(html)
        self.assertEqual(parser.results[0]["url"], "https://example.org/article")
    def test_context_and_reference_urls(self):
        rows = [WebResult("Example", "https://example.org", "A snippet")]
        self.assertIn("[W1] Example", format_web_context(rows))
        self.assertIn("https://example.org", format_web_references(rows))
    def test_empty_query_is_rejected(self):
        from bitnet_windows.web_search import search_web
        with self.assertRaises(ValueError): search_web("  ")

    def test_bing_parser_extracts_result(self):
        from bitnet_windows.web_search import _BingParser
        html = '<li class="b_algo"><h2><a href="https://example.org/page">Example page</a></h2><p>Useful summary</p></li>'
        parser = _BingParser(); parser.feed(html)
        self.assertEqual(parser.results[0]["url"], "https://example.org/page")
        self.assertEqual(parser.results[0]["title"], "Example page")
        self.assertEqual(parser.results[0]["snippet"], "Useful summary")


    def test_deduplicates_same_url_and_keeps_best_snippet(self):
        from bitnet_windows.web_search import deduplicate_results
        rows = [
            {"title": "Forecast", "url": "https://example.org/weather?utm_source=one", "snippet": "Short"},
            {"title": "Forecast duplicate", "url": "http://example.org/weather?utm_source=two", "snippet": "A longer useful weather forecast"},
            {"title": "Other", "url": "https://other.example/page", "snippet": "Other result"},
        ]
        results = deduplicate_results(rows, limit=5)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].snippet, "A longer useful weather forecast")

    def test_deduplicates_references(self):
        from bitnet_windows.web_search import format_web_references
        rows = [
            WebResult("One", "https://example.org/page?utm_source=a", "A"),
            WebResult("Duplicate", "http://example.org/page?utm_source=b", "B"),
        ]
        rendered = format_web_references(rows)
        self.assertEqual(rendered.count("example.org/page"), 1)


    def test_weather_tomorrow_query_includes_exact_date_and_fields(self):
        from datetime import date
        from bitnet_windows.web_search import prepare_search_query
        query = prepare_search_query("Quelle est la météo de demain à Alger ?", today=date(2026, 10, 18))
        self.assertIn("19/10/2026", query)
        self.assertIn("température minimale maximale", query)
        self.assertIn("probabilité de pluie", query)

    def test_non_weather_query_is_not_rewritten(self):
        from datetime import date
        from bitnet_windows.web_search import prepare_search_query
        self.assertEqual(prepare_search_query("histoire d'Alger", today=date(2026, 10, 18)), "histoire d'Alger")

    def test_weather_context_warns_against_unsupported_forecasts(self):
        rows = [WebResult("Prévisions météo Alger", "https://example.org", "Alger météo demain")]
        context = format_web_context(rows)
        self.assertIn("Ne présente pas une estimation inventée", context)
        self.assertIn("date exacte", context)

    def test_search_tries_fallback_after_primary_error(self):
        from unittest.mock import patch
        from bitnet_windows import web_search
        html = '<li class="b_algo"><h2><a href="https://example.org/page">Example page</a></h2><p>Useful summary</p></li>'
        with patch.object(web_search, "_fetch_html", side_effect=[RuntimeError("timed out"), html]):
            results = web_search.search_web("example query")
        self.assertEqual(results[0].url, "https://example.org/page")

if __name__ == "__main__": unittest.main()
