# SPDX-License-Identifier: MIT
# Label rule of the benchmark generator (excerpt). Family identifiers use the paper's names;
# one identifier was renamed (bijective). The audit reads this rule; it never executes the generator.

def norm_str(x) -> str:
    if x is None:
        return ""
    x = str(x).strip().lower()
    x = " ".join(x.split())
    return x

def derive_family(facet: str, term_type: str) -> str:
    facet_n = norm_str(facet)
    tt = norm_str(term_type)

    if tt == "category" or facet_n.startswith("category/"):
        return "taxonomic_category"
    if tt == "part" or facet_n.startswith("part/"):
        return "structural_expression"
    if tt == "color" or facet_n.startswith("color/"):
        return "surface_expression"
    if "nickname" in facet_n or facet_n.startswith("tag/") or tt == "tag":
        return "tag_terms"

    if facet_n.startswith("attribute/"):
        if any(k in facet_n for k in [
            "pattern", "textile", "material", "print", "surface", "leather",
            "non textile material", "finishing", "manufacturing"
        ]):
            return "surface_expression"
        if any(k in facet_n for k in [
            "silhouette", "fit", "length", "neckline", "opening",
            "garment detail", "hem", "sleeve"
        ]):
            return "structural_expression"
        return "attribute_expression"

    if tt == "attribute":
        return "attribute_expression"

    return "other_expression"
