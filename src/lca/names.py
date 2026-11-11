# SPDX-License-Identifier: MIT
"""Canonical names and orders shared by every step.

All matrix columns, arg-max scans, tie handling and CSV orderings use these declared orders,
never an incidental string sort, so renaming a label cannot change a computed value."""

DATASETS = ("DF", "FP")                     # DeepFashion-family and Fashionpedia-family datasets
TASKS = ("facet", "family")
BLOCKS = ("core", "facet", "family", "type", "rep_core", "inst_family", "inst_sem")

# Family identifiers in canonical order (index = family id).
FAMILY_ORDER = (
    "tag_terms",
    "attribute_expression",
    "other_expression",
    "structural_expression",
    "surface_expression",
    "taxonomic_category",
)
# Instance-family labels of the inst_family block in canonical order.
INST_FAMILY_ORDER = (
    "tag_terms",
    "decorative_expression",
    "material_expression",
    "other_expression",
    "structural_expression",
    "surface_expression",
    "taxonomic_category",
)
FAMILY_CODES = {
    "taxonomic_category": "TAX",
    "structural_expression": "STR",
    "surface_expression": "SURF",
    "attribute_expression": "ATTR",
    "tag_terms": "TAG",
}

VERDICT_CODES = {0: "UNVERIFIED", 1: "UNSAFE", 2: "NONDETERMINED"}
