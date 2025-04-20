# pkg imports
import dspy


# local imports
from signatures import InclusionExclusionCriteria, CheckCriteria

class ClassifyByInclusionExclusion(dspy.Module):
    def __init__(self, sr_title: str):
        self.sr_title = sr_title
        self.generate_criteria = dspy.ChainOfThought(InclusionExclusionCriteria)
        self.evaluate_criteria = dspy.ChainOfThought(CheckCriteria)

    def forward(self, citation_title: str, citation_abstract: str):
        criteria = self.generate_criteria(
            systematic_review_title=self.sr_title
        ).criteria
        return self.evaluate_criteria(criteria=criteria,
                                      citation_title=citation_title,
                                      citation_abstract=citation_abstract)
                                      