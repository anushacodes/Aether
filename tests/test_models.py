import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag.config import PipelineConfig
from rag.models import AnswerResult, ChunkRecord, RetrievedHit, SectionRecord, SourceRecord


@pytest.fixture
def examples():
    return json.loads(Path("tests/fixtures/records.json").read_text())


def test_representative_data_contracts(examples):
    for key, record_type in (
        ("source", SourceRecord),
        ("section", SectionRecord),
        ("chunk", ChunkRecord),
        ("hit", RetrievedHit),
        ("answer", AnswerResult),
        ("config", PipelineConfig),
    ):
        record_type.model_validate(examples[key])


def test_unknown_citation_is_rejected(examples):
    answer = copy.deepcopy(examples["answer"])
    answer["citation_ids"] = ["invented-chunk"]
    with pytest.raises(ValidationError, match="citation IDs"):
        AnswerResult.model_validate(answer)


def test_section_cannot_claim_incorrect_raw_length(examples):
    section = copy.deepcopy(examples["section"])
    section["raw_end"] += 1
    with pytest.raises(ValidationError, match="raw range length"):
        SectionRecord.model_validate(section)


def test_source_cannot_supply_arbitrary_download_filename(examples):
    source = copy.deepcopy(examples["source"])
    source["original_filename"] = "../../secrets"
    with pytest.raises(ValidationError):
        SourceRecord.model_validate(source)
