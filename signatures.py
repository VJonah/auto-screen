# pkg imports
import dspy

class Relevance(dspy.Signature):
    """Classify a citation's relevance to a systematic review."""

    sr_title: str = dspy.InputField()
    citation_title: str = dspy.InputField()
    citation_abstract: str = dspy.InputField()
    relevant: bool = dspy.OutputField()
    confidence: float = dspy.OutputField()

class InclusionExclusionCriteria(dspy.Signature):
    """
    Output a set of inclusion/exclusion criteria for the screening of a systematic review.
    """

    systematic_review_title: str = dspy.InputField()
    criteria: list[str] = dspy.OutputField(desc="Inclusion/exclusion criteria and their descrptions.")

class CheckCriteria(dspy.Signature):
    """
    Verify which criteria are satisfied by the title and abstract of a candidate citation.
    """

    criteria: list[str] = dspy.InputField()
    citation_title: str = dspy.InputField()
    citation_abstract: str = dspy.InputField()
    satisfied: list[bool] = dspy.OutputField(desc="Whether each criteria is satisfied or not.")