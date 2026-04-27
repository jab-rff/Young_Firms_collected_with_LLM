from pathlib import Path

from src.aggregation import aggregate_mentions
from src.data_models import Mention, RetrievedItem
from src.mention_extraction import extract_mentions
from src.normalization import normalize_company_name

from recall.candidate_extraction import extract_candidates
from recall.deduplication import deduplicate_candidates
from recall.export import export_candidates
from recall.query_generation import generate_queries


def test_generate_queries_drops_empty_values() -> None:
    assert generate_queries([" ai startups ", "", " climate "]) == ["ai startups", "climate"]


def test_extract_and_deduplicate_candidates() -> None:
    extracted = extract_candidates(["Acme", "", "Beta", "Acme"])
    assert deduplicate_candidates(extracted) == ["Acme", "Beta"]


def test_export_candidates_writes_one_per_line(tmp_path: Path) -> None:
    output_path = tmp_path / "candidates.txt"
    export_candidates(["Acme", "Beta"], output_path)
    assert output_path.read_text(encoding="utf-8") == "Acme\nBeta\n"


def _extract_names(text: str, title: str = "", snippet: str = "") -> set[str]:
    item = RetrievedItem(
        retrieved_item_id="ret_test",
        query_id="q_test",
        query_text="test query",
        source_name="Test Source",
        title=title,
        snippet=snippet,
        url="https://example.com/test",
        language="en",
        retrieved_at="2025-04-27T10:00:00Z",
        raw_text=text,
    )
    return {mention.firm_name_raw for mention in extract_mentions([item])}


def test_extract_mentions_valid_issuu_relocation_context() -> None:
    names = _extract_names("Issuu was founded in Copenhagen and moved headquarters to Palo Alto.")
    assert "Issuu" in names
    assert "Palo Alto" not in names


def test_extract_mentions_valid_netlify_company_context() -> None:
    names = _extract_names("Netlify is a San Francisco-based company founded by Danish founders.")
    assert "Netlify" in names
    assert "San Francisco-based" not in names


def test_extract_mentions_valid_zendesk_hq_move_context() -> None:
    names = _extract_names("Zendesk moved its headquarters from Copenhagen to San Francisco.")
    assert "Zendesk" in names
    assert "San Francisco" not in names


def test_extract_mentions_valid_sitecore_legal_suffixes() -> None:
    names = _extract_names("Sitecore A/S later became Sitecore Corporation.")
    assert "Sitecore A/S" in names
    assert "Sitecore Corporation" in names


def test_extract_mentions_valid_tradeshift_context() -> None:
    names = _extract_names("Tradeshift was founded in Copenhagen and later headquartered in San Francisco.")
    assert "Tradeshift" in names
    assert "San Francisco" not in names


def test_extract_mentions_rejects_san_francisco_based_startup() -> None:
    names = _extract_names("San Francisco-based startup")
    assert "San Francisco-based" not in names
    assert "San Francisco" not in names


def test_extract_mentions_rejects_united_states_fragment() -> None:
    names = _extract_names("United States. The company")
    assert "United States" not in names
    assert "United States. The" not in names


def test_extract_mentions_rejects_about_and_why_title_fragments() -> None:
    assert "About Netlify" not in _extract_names("About Netlify", title="About Netlify")
    assert "Why Sitecore" not in _extract_names("Why Sitecore", title="Why Sitecore")


def test_extract_mentions_rejects_generic_single_word() -> None:
    assert "Technologies" not in _extract_names("Technologies")


def test_extract_mentions_rejects_san_francisco_headquartered_phrase() -> None:
    names = _extract_names("San Francisco headquartered")
    assert "San Francisco headquartered" not in names
    assert "San Francisco" not in names


def test_extract_mentions_rejects_person_and_keeps_company() -> None:
    names = _extract_names("Mikkel Svane founded Zendesk")
    assert "Mikkel Svane" not in names
    assert "Zendesk" in names


def test_extract_mentions_keeps_issuu_not_palo_alto() -> None:
    item = RetrievedItem(
        retrieved_item_id="ret_issuu",
        query_id="q_001",
        query_text="issuu palo alto",
        source_name="Test Source",
        title="",
        snippet="",
        url="https://example.com/issuu",
        language="en",
        retrieved_at="2025-04-27T10:00:00Z",
        raw_text="Issuu was founded in Copenhagen and moved headquarters to Palo Alto.",
    )

    names = {mention.firm_name_raw for mention in extract_mentions([item])}

    assert "Issuu" in names
    assert "Palo Alto" not in names


def test_extract_mentions_keeps_tradeshift_not_san_francisco() -> None:
    item = RetrievedItem(
        retrieved_item_id="ret_tradeshift",
        query_id="q_002",
        query_text="tradeshift san francisco",
        source_name="Test Source",
        title="",
        snippet="",
        url="https://example.com/tradeshift",
        language="en",
        retrieved_at="2025-04-27T10:05:00Z",
        raw_text="Tradeshift later headquartered in San Francisco.",
    )

    names = {mention.firm_name_raw for mention in extract_mentions([item])}

    assert "Tradeshift" in names
    assert "San Francisco" not in names


def test_extract_mentions_rejects_sentence_boundary_fragments() -> None:
    item = RetrievedItem(
        retrieved_item_id="ret_fragment",
        query_id="q_003",
        query_text="united states fragment",
        source_name="Test Source",
        title="",
        snippet="",
        url="https://example.com/fragment",
        language="en",
        retrieved_at="2025-04-27T10:10:00Z",
        raw_text="United States. The company later expanded abroad.",
    )

    names = {mention.firm_name_raw for mention in extract_mentions([item])}

    assert "United States" not in names
    assert "United States. The" not in names


def test_extract_mentions_rejects_location_adjective() -> None:
    item = RetrievedItem(
        retrieved_item_id="ret_based",
        query_id="q_004",
        query_text="san francisco based",
        source_name="Test Source",
        title="",
        snippet="",
        url="https://example.com/based",
        language="en",
        retrieved_at="2025-04-27T10:15:00Z",
        raw_text="The startup later became San Francisco-based after fundraising.",
    )

    names = {mention.firm_name_raw for mention in extract_mentions([item])}

    assert "San Francisco-based" not in names
    assert "San Francisco" not in names


def test_extract_mentions_rejects_newline_spanning_matches() -> None:
    item = RetrievedItem(
        retrieved_item_id="ret_newlines",
        query_id="q_004b",
        query_text="newline artifacts",
        source_name="Test Source",
        title="Issuu",
        snippet="Issuu",
        url="https://example.com/newlines",
        language="en",
        retrieved_at="2025-04-27T10:16:00Z",
        raw_text="Issuu\nIssuu\nSan Francisco\nHistorien\nSan Francisco\nPodio\nMagma\nArtiklen",
    )

    names = {mention.firm_name_raw for mention in extract_mentions([item])}

    assert "Issuu\nIssuu" not in names
    assert "San Francisco\nHistorien" not in names
    assert "San Francisco\nPodio" not in names
    assert "Magma\nArtiklen" not in names


def test_extract_mentions_rejects_about_title_fragment() -> None:
    item = RetrievedItem(
        retrieved_item_id="ret_about",
        query_id="q_005",
        query_text="about netlify",
        source_name="Test Source",
        title="About Netlify",
        snippet="About Netlify and its platform.",
        url="https://example.com/about-netlify",
        language="en",
        retrieved_at="2025-04-27T10:20:00Z",
        raw_text="About Netlify and its developer platform.",
    )

    names = {mention.firm_name_raw for mention in extract_mentions([item])}

    assert "About Netlify" not in names


def test_extract_mentions_rejects_headline_fragment() -> None:
    item = RetrievedItem(
        retrieved_item_id="ret_headline",
        query_id="q_006",
        query_text="zendesk expands san francisco presence",
        source_name="Test Source",
        title="Zendesk Expands San Francisco Presence",
        snippet="Zendesk Expands San Francisco Presence",
        url="https://example.com/zendesk-headline",
        language="en",
        retrieved_at="2025-04-27T10:25:00Z",
        raw_text="Zendesk Expands San Francisco Presence.",
    )

    names = {mention.firm_name_raw for mention in extract_mentions([item])}

    assert "Zendesk Expands San Francisco Presence" not in names


def test_extract_mentions_rejects_generic_single_words() -> None:
    item = RetrievedItem(
        retrieved_item_id="ret_generic",
        query_id="q_006b",
        query_text="generic words",
        source_name="Test Source",
        title="Technologies Software Company Startup Article Artiklen Historien Profilen",
        snippet="",
        url="https://example.com/generic",
        language="en",
        retrieved_at="2025-04-27T10:26:00Z",
        raw_text="Technologies Software Company Startup Article Artiklen Historien Profilen",
    )

    names = {mention.firm_name_raw for mention in extract_mentions([item])}

    for rejected in {
        "Technologies",
        "Software",
        "Company",
        "Startup",
        "Article",
        "Artiklen",
        "Historien",
        "Profilen",
    }:
        assert rejected not in names


def test_extract_mentions_rejects_sample_founder_names() -> None:
    item = RetrievedItem(
        retrieved_item_id="ret_people",
        query_id="q_006c",
        query_text="founder names",
        source_name="Test Source",
        title="",
        snippet="",
        url="https://example.com/people",
        language="en",
        retrieved_at="2025-04-27T10:27:00Z",
        raw_text=(
            "Alexander Aghassipour and Mikkel Svane worked with Michael Hansen. "
            "Anders Pollas also joined Christian Bach."
        ),
    )

    names = {mention.firm_name_raw for mention in extract_mentions([item])}

    for rejected in {
        "Alexander Aghassipour",
        "Mikkel Svane",
        "Michael Hansen",
        "Anders Pollas",
        "Christian Bach",
    }:
        assert rejected not in names


def test_extract_mentions_keeps_valid_one_word_firms_in_context() -> None:
    item = RetrievedItem(
        retrieved_item_id="ret_valid_one_word",
        query_id="q_006d",
        query_text="valid one word firms",
        source_name="Test Source",
        title="",
        snippet="",
        url="https://example.com/valid-one-word",
        language="en",
        retrieved_at="2025-04-27T10:28:00Z",
        raw_text=(
            "Issuu was founded in Copenhagen. "
            "Netlify was founded in Denmark. "
            "Podio moved headquarters abroad. "
            "Zendesk later headquartered in San Francisco. "
            "Tradeshift later headquartered in London. "
            "Sitecore relocated headquarters to the United States."
        ),
    )

    names = {mention.firm_name_raw for mention in extract_mentions([item])}

    for expected in {"Issuu", "Netlify", "Podio", "Zendesk", "Tradeshift", "Sitecore"}:
        assert expected in names


def test_aggregate_mentions_merges_issuu_legal_suffix_variants() -> None:
    mentions = [
        Mention(
            mention_id="ment_issuu_1",
            retrieved_item_id="ret_issuu_1",
            query_id="q_007",
            query_text="issuu",
            source_name="Wikipedia",
            title="Issuu",
            url="https://example.com/issuu-1",
            language="en",
            retrieved_at="2025-04-27T10:30:00Z",
            firm_name_raw="Issuu Inc",
            normalized_name=normalize_company_name("Issuu Inc"),
            evidence_text="Issuu Inc moved abroad.",
        ),
        Mention(
            mention_id="ment_issuu_2",
            retrieved_item_id="ret_issuu_2",
            query_id="q_008",
            query_text="issuu",
            source_name="Tech News",
            title="Issuu",
            url="https://example.com/issuu-2",
            language="en",
            retrieved_at="2025-04-27T10:35:00Z",
            firm_name_raw="Issuu",
            normalized_name=normalize_company_name("Issuu"),
            evidence_text="Issuu was founded in Copenhagen.",
        ),
    ]

    candidates = aggregate_mentions(mentions)

    assert len(candidates) == 1
    assert candidates[0].normalized_name == "issuu"
    assert set(candidates[0].raw_name_variants) == {"Issuu", "Issuu Inc"}


def test_aggregate_mentions_merges_sitecore_legal_suffix_variants() -> None:
    mentions = [
        Mention(
            mention_id="ment_sitecore_1",
            retrieved_item_id="ret_sitecore_1",
            query_id="q_009",
            query_text="sitecore",
            source_name="Wikipedia",
            title="Sitecore",
            url="https://example.com/sitecore-1",
            language="en",
            retrieved_at="2025-04-27T10:40:00Z",
            firm_name_raw="Sitecore A/S",
            normalized_name=normalize_company_name("Sitecore A/S"),
            evidence_text="Sitecore A/S was founded in Denmark.",
        ),
        Mention(
            mention_id="ment_sitecore_2",
            retrieved_item_id="ret_sitecore_2",
            query_id="q_010",
            query_text="sitecore corporation",
            source_name="Business News",
            title="Sitecore Corporation",
            url="https://example.com/sitecore-2",
            language="en",
            retrieved_at="2025-04-27T10:45:00Z",
            firm_name_raw="Sitecore Corporation",
            normalized_name=normalize_company_name("Sitecore Corporation"),
            evidence_text="Sitecore Corporation is now headquartered abroad.",
        ),
    ]

    candidates = aggregate_mentions(mentions)

    assert len(candidates) == 1
    assert candidates[0].normalized_name == "sitecore"
    assert set(candidates[0].raw_name_variants) == {"Sitecore A/S", "Sitecore Corporation"}
