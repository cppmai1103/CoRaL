# SEA-NLI (aisingapore/SEA-NLI, configs `normal` and `hard`): 3-way NLI, one test split for all cultures. The
# `culture` values are lowercase adjectives (checked: myanmar, filipino, indonesian, cambodian, malaysian, thai,
# vietnamese, plus singaporean, which is not a project language), not the country names in
# docs/03b_additional_benchmarks.md. The gold label is `true_label`.
LABELS = ["entailment", "contradiction", "neutral"]


def sea_choices(doc):
    return LABELS


def sea_target(doc):
    label = str(doc["true_label"]).strip().lower()
    if label not in LABELS:
        raise ValueError(f"Unknown SEA-NLI label: {label}")
    return LABELS.index(label)


def _filter_culture(dataset, culture):
    return dataset.filter(lambda row: row["culture"] == culture)


def filter_myanmar(dataset):
    return _filter_culture(dataset, "myanmar")


def filter_filipino(dataset):
    return _filter_culture(dataset, "filipino")


def filter_indonesian(dataset):
    return _filter_culture(dataset, "indonesian")


def filter_cambodian(dataset):
    return _filter_culture(dataset, "cambodian")


def filter_malaysian(dataset):
    return _filter_culture(dataset, "malaysian")


def filter_thai(dataset):
    return _filter_culture(dataset, "thai")


def filter_vietnamese(dataset):
    return _filter_culture(dataset, "vietnamese")
