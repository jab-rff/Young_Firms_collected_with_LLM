from src.diffbot_founder_pipeline import build_diffbot_candidate, build_diffbot_paths


def test_build_diffbot_candidate_keeps_diffbot_as_hint_not_truth() -> None:
    row = {
        "name": "Zendesk",
        "founders_name": "Alexander Aghassipour,Morten Primdahl,Mikkel Svane",
        "homepageUri": "zendesk.com",
        "linkedInUri": "linkedin.com/company/418095",
        "location_city_name": "San Francisco",
        "location_region_name": "California",
        "location_country_name": "United States",
        "summary": "Software company based in San Francisco, California",
        "categories_name": "Software Companies",
        "ceo_name": "Tom Eggemeier",
        "nbEmployeesMax": 20000,
        "founders_targetDiffbotId": "abc,def",
    }

    candidate = build_diffbot_candidate(row)

    assert candidate["firm_name"] == "Zendesk"
    assert candidate["origin_track"] == "abroad_danish_founders"
    assert candidate["founded_in_denmark"] == "uncertain"
    assert candidate["danish_founders_abroad"] == "uncertain"
    assert candidate["moved_hq_abroad"] == "uncertain"
    assert candidate["sources"] == ["zendesk.com", "linkedin.com/company/418095"]
    assert "unverified hints" in candidate["confidence_note"].lower()
    assert candidate["third_party_seed"]["founders_name"].startswith("Alexander")


def test_build_diffbot_paths_are_separate_from_snowball_outputs() -> None:
    paths = build_diffbot_paths()

    assert "data/diffbot" in str(paths.candidates).replace("\\", "/")
    assert paths.candidates.name.endswith("_candidates.jsonl")
    assert paths.model2.name.endswith("_enriched.jsonl")
    assert paths.model3.name.endswith("_validated.jsonl")
    assert paths.review.name.endswith("_review.csv")
