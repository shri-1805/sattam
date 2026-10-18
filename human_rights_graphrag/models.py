"""Typed domain and API models."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ScenarioRequest(BaseModel):
    """A human-rights scenario submitted for analysis."""

    scenario: str = Field(min_length=1, max_length=12_000)

    @field_validator("scenario")
    @classmethod
    def scenario_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("scenario must not be blank")
        return value.strip()


class SubClauseContext(BaseModel):
    """A provision nested within an article."""

    id: str
    text: str


class ArticleContext(BaseModel):
    """A retrieved article and its connected graph context."""

    id: str
    document: str
    reference: str
    title: str
    text: str
    score: float
    subclauses: list[SubClauseContext] = Field(default_factory=list)
    rights: list[str] = Field(default_factory=list)


class SelectedClause(BaseModel):
    """A retrieved clause selected by the language model."""

    model_config = ConfigDict(extra="forbid")

    id: str
    applicability: str = Field(min_length=1, max_length=2000)


class ModelSelection(BaseModel):
    """The constrained response expected from the language model."""

    model_config = ConfigDict(extra="forbid")

    clauses: list[SelectedClause]


class ClauseResponse(BaseModel):
    """Verbatim retrieved legal text with a tailored explanation."""

    id: str
    article_id: str
    document: str
    reference: str
    title: str
    text: str
    applicability: str
    rights: list[str]


class AnalysisResponse(BaseModel):
    """Human-rights analysis response."""

    scenario: str
    clauses: list[ClauseResponse]
    notice: str = (
        "Informational only; this is not legal advice. Verify the current law "
        "and consult a qualified lawyer for advice."
    )
