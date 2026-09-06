import unittest

from utils.research_rag import _report_metadata, _split_chunks


class ResearchRagParsingTests(unittest.TestCase):
	def test_report_metadata_extracts_title_query_and_sources(self) -> None:
		title, query, sources = _report_metadata(
			"# Deep Research: Black Metal\n\n"
			"- Original request: deep research: black metal\n\n"
			"## Sources\n\n1. https://example.com/source\n"
		)
		self.assertEqual(title, "Black Metal")
		self.assertEqual(query, "deep research: black metal")
		self.assertEqual(sources, ["https://example.com/source"])

	def test_split_chunks_preserves_sections_and_bounds_large_text(self) -> None:
		chunks = _split_chunks("# Report\n\n## First\n\n" + "word " * 900)
		self.assertGreater(len(chunks), 1)
		self.assertTrue(all(section == "First" for section, _ in chunks))
		self.assertTrue(all(len(content) <= 3500 for _, content in chunks))


if __name__ == "__main__":
	unittest.main()
