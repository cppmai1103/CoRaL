# SIB-200 (Davlan/sib200): 7-way topic classification. The dataset stores the topic as a string in `category`
# (not an integer `label` as docs/03b_additional_benchmarks.md assumed); the choice order below is fixed and every
# category value of the 7 project languages was checked against it.
TOPICS = ["science/technology", "travel", "politics", "sports", "health", "entertainment", "geography"]


def sib_choices(doc):
    return TOPICS


def sib_target(doc):
    return TOPICS.index(doc["category"])
