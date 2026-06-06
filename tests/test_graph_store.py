import json
from datetime import datetime, timezone

import pytest

from prism import ClaimStatus, ClaimTriple
from prism.memory import MemoryGraphStore


NOW = datetime(2026, 6, 6, 18, 0, tzinfo=timezone.utc)


def make_claim(
    *,
    subject: str = "PRISM",
    relation: str = "is",
    object_: str = "prototype",
    claim_id: str = "claim-1",
    status: ClaimStatus | str = ClaimStatus.PROPOSED,
) -> ClaimTriple:
    return ClaimTriple(
        subject=subject,
        relation=relation,
        object=object_,
        confidence=0.87,
        source="unit-test",
        timestamp=NOW,
        status=status,
        claim_id=claim_id,
        run_id="run-1",
        turn_id="turn-1",
        provenance={"extractor": "manual", "span": "PRISM is a prototype."},
    )


def test_creating_empty_graph_store() -> None:
    store = MemoryGraphStore()

    assert len(store) == 0
    assert store.is_empty
    assert store.claims == ()
    assert store.to_dict()["claims"] == []


def test_adding_one_claim() -> None:
    store = MemoryGraphStore()
    claim = make_claim()

    added = store.add_claim(claim)

    assert added is claim
    assert len(store) == 1
    assert store.claims == (claim,)


def test_retrieving_claims_by_subject() -> None:
    store = MemoryGraphStore(
        [
            make_claim(subject="PRISM", object_="prototype", claim_id="claim-1"),
            make_claim(subject="Semantic Entropy", object_="uncertainty metric", claim_id="claim-2"),
        ]
    )

    matches = store.get_claims_by_subject("  prism ")

    assert [claim.claim_id for claim in matches] == ["claim-1"]


def test_retrieving_claims_by_entity_checks_subject_and_object() -> None:
    store = MemoryGraphStore(
        [
            make_claim(subject="PRISM", object_="prototype", claim_id="claim-1"),
            make_claim(subject="Semantic Entropy", object_="PRISM", claim_id="claim-2"),
            make_claim(subject="SelfCheckGPT", object_="baseline", claim_id="claim-3"),
        ]
    )

    matches = store.get_claims_by_entity("prism")

    assert [claim.claim_id for claim in matches] == ["claim-1", "claim-2"]


def test_retrieving_claims_by_relation() -> None:
    store = MemoryGraphStore(
        [
            make_claim(relation="uses", object_="claim triples", claim_id="claim-1"),
            make_claim(relation="is", object_="prototype", claim_id="claim-2"),
        ]
    )

    matches = store.get_claims_by_relation("USES")

    assert [claim.claim_id for claim in matches] == ["claim-1"]


def test_duplicate_claim_handling_returns_existing_without_overwrite() -> None:
    store = MemoryGraphStore()
    original = make_claim(subject="  PRISM", relation="is", object_="prototype", claim_id="claim-1")
    duplicate = ClaimTriple(
        subject="prism",
        relation=" is ",
        object="  prototype ",
        confidence=0.2,
        source="later-source",
        timestamp=NOW,
        claim_id="claim-duplicate",
        run_id="run-2",
        turn_id="turn-2",
        provenance={"extractor": "different"},
    )

    first = store.add_claim(original)
    second = store.add_claim(duplicate)

    assert first is original
    assert second is original
    assert len(store) == 1
    assert store.find_duplicate(duplicate) is original
    assert store.claims[0].source == "unit-test"
    assert store.claims[0].run_id == "run-1"


@pytest.mark.parametrize(
    "status",
    [
        ClaimStatus.PROPOSED,
        ClaimStatus.ACCEPTED,
        ClaimStatus.REJECTED,
        ClaimStatus.CONTRADICTED,
    ],
)
def test_status_update(status: ClaimStatus) -> None:
    store = MemoryGraphStore([make_claim()])

    updated = store.update_claim_status("claim-1", status)

    assert updated.status is status
    assert store.claims[0].status is status


def test_export_to_json(tmp_path) -> None:
    store = MemoryGraphStore([make_claim()])
    path = tmp_path / "memory.json"

    store.export_json(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["claims"][0]["claim_id"] == "claim-1"


def test_import_from_json(tmp_path) -> None:
    original = MemoryGraphStore([make_claim()])
    path = tmp_path / "memory.json"
    original.export_json(path)

    restored = MemoryGraphStore.import_json(path)

    assert len(restored) == 1
    assert restored.claims[0].canonical_key == ("prism", "is", "prototype")


def test_provenance_preservation_through_json_round_trip() -> None:
    store = MemoryGraphStore([make_claim()])

    restored = MemoryGraphStore.from_json(store.to_json())
    claim = restored.claims[0]

    assert claim.confidence == 0.87
    assert claim.source == "unit-test"
    assert claim.timestamp == NOW
    assert claim.run_id == "run-1"
    assert claim.turn_id == "turn-1"
    assert claim.provenance == {"extractor": "manual", "span": "PRISM is a prototype."}
