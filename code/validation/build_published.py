"""Published ICILS 2023 CIL means / gender gaps / participation flags vs computed values.

All published values were transcribed by hand (tables are vector graphics in the PDF)
from the official sources below; nothing is imputed or derived.

  [IR] Fraillon, J. (Ed.) (2025). An International Perspective on Digital Literacy:
       Results from ICILS 2023. Springer. doi:10.1007/978-3-031-87722-3
       - Table 5.1 "Country averages and distribution for CIL", printed p.108 (PDF p.129)
       - Table 6.1 "CIL achievement by gender", printed p.122 (PDF p.143)
       - Table J.2 "Netherlands's average and distribution for CIL and CT", p.321 (PDF p.334)
       - Table J.5 "CIL and CT achievement by gender in the Netherlands", p.322 (PDF p.335)
       - Table 1.1 footnotes a/b, p.3 (PDF p.26)
  [TR] IEA (2026). ICILS 2023 Technical Report (version of 3 July 2026).
       https://www.iea.nl/sites/default/files/2026-06/ICILS_2023_Technical_Report.pdf
       - Table 7.4 weighted teacher PR, p.103 (PDF p.121)
       - Table 7.5 annotations, p.106 (PDF p.124)
       - Table 7.6 reporting category, student survey, pp.107-108 (PDF pp.125-126)
       - Table 7.7 reporting category, teacher survey, pp.109-111 (PDF pp.127-129)
  [IR-A2] Table A.2 student participation (weighted overall PR after replacement), p.205 (PDF p.225)

Usage (from code/): python -I validation/build_published.py <repository root, e.g. ..>
Needs results/validation_country_means.csv, written by 02_validate.py.
"""
import csv
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
sys.path.insert(0, str(ROOT / "code"))
from countries import CNAME  # noqa: E402

SRC_MEAN = "IR Table 5.1, p.108 (PDF p.129)"
SRC_GAP = "IR Table 6.1, p.122 (PDF p.143)"

# code: (mean, se, Table-5.1 annotation symbols)
T51 = {
    "KOR": (540, 2.5, "†"), "CZE": (525, 2.1, "1"), "DNK": (518, 2.7, "†1"),
    "TWN": (515, 3.0, ""), "BFL": (511, 4.4, "†"), "PRT": (510, 3.0, "1"),
    "LVA": (509, 3.6, "1"), "FIN": (507, 3.6, ""), "AUT": (506, 2.5, "1"),
    "HUN": (505, 3.8, ""), "SWE": (504, 3.0, "1"), "NOR": (502, 2.9, "1"),
    "DEU": (502, 3.5, ""), "SVK": (499, 2.7, ""), "FRA": (498, 2.7, ""),
    "ESP": (495, 1.9, "1"), "LUX": (494, 2.0, ""), "ITA": (491, 2.6, ""),
    "HRV": (487, 3.9, "1"), "SVN": (483, 2.3, "1"), "MLT": (475, 2.5, ""),
    "CYP": (460, 2.6, ""), "GRC": (460, 3.3, ""), "URY": (447, 3.6, "†"),
    "SRB": (443, 3.7, "1"), "BIH": (440, 3.8, "3"), "ROU": (418, 5.3, "†12"),
    "KAZ": (407, 3.1, "1"), "OMN": (379, 3.0, ""), "XKX": (356, 4.1, "1"),
    "AZE": (319, 5.1, ""), "DNW": (485, 4.1, "1"), "USA": (482, 6.6, "‡"),
}
# ICILS 2023 average (Table 5.1): 476 (0.6)

# code: (female - male, se)  Table 6.1
T61 = {
    "OMN": (53, 5.9), "HRV": (37, 5.1), "MLT": (32, 4.2), "TWN": (30, 3.3),
    "KOR": (29, 3.9), "SVN": (27, 2.9), "CYP": (26, 4.3), "NOR": (26, 3.7),
    "FIN": (24, 3.8), "DNK": (23, 3.6), "LVA": (22, 3.7), "AZE": (20, 4.4),
    "LUX": (19, 3.1), "ESP": (19, 2.5), "ITA": (18, 2.6), "BFL": (16, 4.9),
    "SWE": (16, 4.0), "AUT": (15, 3.3), "GRC": (15, 3.9), "KAZ": (15, 3.3),
    "XKX": (14, 4.5), "BIH": (13, 5.7), "ROU": (12, 5.9), "SRB": (11, 4.0),
    "DEU": (10, 3.7), "PRT": (9, 3.5), "SVK": (8, 2.8), "FRA": (8, 2.9),
    "HUN": (6, 3.7), "URY": (6, 4.2), "CZE": (3, 1.8), "DNW": (6, 6.5),
    "USA": (25, 6.0),
}
GAP_NS = {"HUN", "URY", "CZE", "DNW"}  # not bold in Table 6.1 (p >= .05)
# ICILS 2023 average gap (Table 6.1): 19 (0.7)

SYM = {
    "†": "† Met guidelines for sampling participation rates only after replacement schools were included.",
    "‡": "‡ Did not meet guideline for sampling participation rate, but achieved at least 50% overall sampling participation rate.",
    "1": "1 National defined population covers 90% to 95% of the national target population.",
    "2": "2 Country surveyed target grade in the first half of the school year.",
    "3": "3 National defined population covers 61% of the national target population.",
}

# ---- participation flags: TR Table 7.6 (students) and 7.7 (teachers) ----
D = "† Met guidelines for sampling participation rates only after replacement schools were included."
DD = "‡ Does not meet guideline for sampling participation rate, but achieved at least 50% overall sampling participation rate."
NR = "Not reported: Does not meet guideline for sampling participation rate for reporting."
C1 = "Category 1 (met requirements; no participation annotation)"
N1 = "1 National defined population covers 90% to 95% of the national target population."
N2 = "2 Country surveyed target grade in the first half of the school year."
N3 = "3 National defined population covers 61% of the national target population."
PSI = ("Sampling procedures for teacher listing were not followed and any conclusions about teacher analysis "
       "is subject to caution (possible biases in both point estimates and standard errors). "
       "[footnote **: Participation rates were of Category 1, however not following sampling procedures for "
       "teacher listing might have introduced bias into the data.]")
OSS = ("[footnote ***: Due to the OSS server hack, some teacher data got lost and couldn't be recovered ... "
       "teacher participation rates for these countries are assumed to be higher than reported.]")
CHL_S = ("Not reported as requested by country, due to additional national data collection in school year "
         "2023/2024. Data would be reported in Category 3.")
CHL_T = ("Not reported as requested by country, due to additional national data collection in school year "
         "2023/2024. Data does not meet guideline for sampling participation rate for reporting.")
STU = {
    "AUT": f"{C1}; {N1}", "AZE": C1, "BFL": f"Category 2; {D}", "BIH": f"{C1}; {N3}",
    "CHL": f"Not reported; {CHL_S}", "TWN": C1, "HRV": f"{C1}; {N1}", "CYP": C1,
    "CZE": f"{C1}; {N1}", "DNK": f"Category 2; {D}; {N1}", "FIN": C1, "FRA": C1, "DEU": C1,
    "GRC": C1, "HUN": f"{C1} [footnote *: Hungary did not apply double coding for occupations - not annotated]",
    "ITA": C1, "KAZ": f"{C1}; {N1}", "KOR": f"Category 2; {D}", "XKX": f"{C1}; {N1}",
    "LVA": f"{C1}; {N1}", "LUX": C1, "MLT": C1, "NLD": NR, "NOR": f"{C1}; {N1}", "OMN": C1,
    "PRT": f"{C1}; {N1}", "ROU": f"Category 2; {D}; {N1}; {N2}", "SRB": f"{C1}; {N1}", "SVK": C1,
    "SVN": f"{C1}; {N1}", "ESP": f"{C1}; {N1}", "SWE": f"{C1}; {N1}", "USA": f"Category 3; {DD}",
    "URY": f"Category 2; {D}", "DNW": f"{C1} (benchmarking participant)",
}
TEA = {
    "AUT": f"Category 3; {DD}", "AZE": C1, "BFL": f"Category 3; {DD}", "BIH": NR,
    "CHL": f"Not reported; {CHL_T}", "TWN": C1, "HRV": C1, "CYP": f"Category 3; {DD}", "CZE": C1,
    "DNK": f"Category 3; {DD}", "FIN": C1, "FRA": f"Category 3; {DD}", "DEU": f"Category 3; {DD}",
    "GRC": (f"Category 2; {D} [footnote *: Greece included also teachers who joined within the first two "
            "month after the beginning of the school year - not annotated]"),
    "HUN": C1, "ITA": C1, "KAZ": C1, "KOR": C1, "XKX": f"Category 3; {DD}",
    "LVA": f"Category 2; {D}; {N1}", "LUX": NR, "MLT": f"Category 3; {DD}", "NLD": NR,
    "NOR": f"{C1} {OSS}", "OMN": f"Category 3; Ψ {PSI}", "PRT": f"{C1}; {N1}",
    "ROU": f"Category 2; {D}; {N2}", "SRB": C1, "SVK": f"Category 3; Ψ {PSI}", "SVN": C1, "ESP": C1,
    "SWE": f"Category 3; {DD} {OSS}", "USA": NR, "URY": f"Category 3; {DD}",
    "DNW": f"{C1} (benchmarking participant)",
}
# TR Table 7.4: weighted overall teacher PR after replacement (%)
TPR = {"AUT": 72.3, "AZE": 80.6, "BFL": 61.1, "BIH": 28.8, "TWN": 90.6, "HRV": 86.5, "CYP": 71.6,
       "CZE": 95.0, "DNK": 63.0, "FIN": 89.5, "FRA": 66.1, "DEU": 57.5, "GRC": 81.1, "HUN": 76.1,
       "ITA": 96.7, "KAZ": 99.8, "KOR": 95.1, "XKX": 53.6, "LVA": 78.1, "LUX": 28.8, "MLT": 62.0,
       "NLD": 13.9, "NOR": 76.1, "OMN": 94.8, "PRT": 88.0, "ROU": 77.8, "SRB": 92.1, "SVK": 86.7,
       "SVN": 88.6, "ESP": 86.7, "SWE": 74.1, "USA": 47.7, "URY": 68.5, "DNW": 80.5}
# IR Table A.2: weighted overall student PR after replacement (%)
SPR = {"AUT": 92.3, "AZE": 92.6, "BFL": 82.8, "BIH": 89.5, "TWN": 95.6, "HRV": 89.7, "CYP": 90.5,
       "CZE": 94.5, "DNK": 82.1, "FIN": 91.9, "FRA": 92.9, "DEU": 85.2, "GRC": 89.9, "HUN": 92.7,
       "ITA": 96.5, "KAZ": 96.4, "KOR": 95.3, "XKX": 94.2, "LVA": 85.0, "LUX": 85.8, "MLT": 90.3,
       "NLD": 25.9, "NOR": 82.4, "OMN": 96.1, "PRT": 95.0, "ROU": 78.7, "SRB": 92.5, "SVK": 92.9,
       "SVN": 89.5, "ESP": 90.7, "SWE": 83.8, "USA": 56.0, "URY": 78.2, "DNW": 88.9}
SRC_PART = ("TR Table 7.6 pp.107-108 (PDF 125-126) [students]; TR Table 7.7 pp.109-111 (PDF 127-129) "
            "[teachers]; annotations TR Table 7.5 p.106")


def fmt(x):
    return "NA" if x is None else x


def main():
    rows = []
    for code, name in sorted(CNAME.items(), key=lambda kv: kv[1]):
        mean = se = gap = gse = None
        src, flags = [], []
        if code in T51:
            mean, se, sym = T51[code]
            src.append(SRC_MEAN)
            for s in ["†", "‡"] + list("123"):
                if s in sym:
                    flags.append(f"Table 5.1 {SYM[s]}")
            if code == "DNW":
                flags.append("Benchmarking participant (listed separately; not in ICILS 2023 average)")
            if code == "ROU":
                flags.append("Not included in ICILS 2023 average (Table 5.1/6.1 notes: average based on all "
                             "non-benchmarking participants that met sampling participation requirements except Romania)")
            if code == "USA":
                flags.append("Listed under 'Country not meeting sample participation requirements'; not in ICILS 2023 average")
            if code == "NOR":
                flags.append("Reported as 'Norway (Grade 9)'")
        if code in T61:
            gap, gse = T61[code]
            src.append(SRC_GAP)
            if code in GAP_NS:
                flags.append("Table 6.1 gender difference not statistically significant (p>=.05)")
        if code == "NLD":
            mean, se = 460, 10.1
            src.append("IR Table J.2, p.321 (PDF p.334)")
            flags.append("Netherlands excluded from main report tables (Table 1.1 note b: issues with main survey data "
                         "collection; selected data in Appendix J); student overall PR after replacement 25.9% (Table A.2); "
                         "Table J.5 p.322 reports CIL male 446 (11.0), female 475 (10.2) but no published difference/SE -> gap NA")
        if code == "CHL":
            src.append("IR Table 1.1 note a, p.3 (PDF p.26)")
            flags.append("Chile not reported in international report (Table 1.1 note a: issues with main survey data "
                         "collection; additional national data collection) -> all values NA")
        rows.append({
            "CNTRY": code, "system_name": name,
            "published_mean": fmt(mean), "published_se": fmt(se),
            "published_gender_gap": fmt(gap), "published_gender_se": fmt(gse),
            "source": "; ".join(src) if src else "NA",
            "flags": " | ".join(flags),
            "student_participation_flag": STU.get(code, "NA"),
            "teacher_participation_flag": TEA.get(code, "NA"),
            "student_overall_PR_after_repl_w": fmt(SPR.get(code)),
            "teacher_overall_PR_after_repl_w": fmt(TPR.get(code)),
            "participation_source": SRC_PART + "; student PR: IR Table A.2 p.205 (PDF 225); teacher PR: TR Table 7.4 p.103 (PDF 121)",
        })
    out = ROOT / "results"
    with open(out / "published_cil_means.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- computed values (02_validate.py), at the one-decimal precision of the comparison ----
    comp = {}
    with open(out / "validation_country_means.csv", newline="", encoding="utf-8") as f:
        for v in csv.DictReader(f):
            comp[v["CNTRY"]] = tuple(float(f"{float(v[k]):.1f}") for k in
                                     ("CIL_mean", "CIL_se", "gap_girls_minus_boys", "gap_se"))
    cmp_rows = []
    for r in rows:
        c = r["CNTRY"]
        if c not in comp:
            continue
        cm, cse, cg, cgse = comp[c]
        pm, pse, pg, pgse = (r[k] for k in ("published_mean", "published_se", "published_gender_gap", "published_gender_se"))
        d = {"CNTRY": c, "system_name": r["system_name"], "computed_mean": cm, "published_mean": pm,
             "diff_mean": "NA", "computed_se": cse, "published_se": pse, "se_ratio": "NA",
             "computed_gap": cg, "published_gap": pg, "diff_gap": "NA",
             "computed_gap_se": cgse, "published_gap_se": pgse, "gap_se_ratio": "NA", "note": ""}
        notes = []
        if pm != "NA":
            d["diff_mean"] = round(cm - pm, 1)
            d["se_ratio"] = round(cse / pse, 3)
            if abs(d["diff_mean"]) > 1:
                notes.append("|diff mean|>1")
            if not 0.9 <= d["se_ratio"] <= 1.1:
                notes.append("SE ratio outside 0.9-1.1")
        if pg != "NA":
            d["diff_gap"] = round(cg - pg, 1)
            d["gap_se_ratio"] = round(cgse / pgse, 3)
            if abs(d["diff_gap"]) > 1:
                notes.append("|diff gap|>1")
            if not 0.9 <= d["gap_se_ratio"] <= 1.1:
                notes.append("gap SE ratio outside 0.9-1.1")
        if pm == "NA":
            notes.append("no published value")
        d["note"] = "; ".join(notes) + ("" if not notes else "") + \
            ("; published means/gaps are integers, SEs 1 decimal (rounding tolerance 0.5 / 0.05)" if pm != "NA" else "")
        d["note"] = d["note"].lstrip("; ")
        cmp_rows.append(d)
    with open(out / "published_vs_computed.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(cmp_rows[0]))
        w.writeheader()
        w.writerows(cmp_rows)

    def summ(key, absval=True):
        v = [abs(x[key]) if absval else x[key] for x in cmp_rows if x[key] != "NA"]
        return len(v), st.median(v), max(v), min(v)
    print("computed systems:", len(comp), "| matched with published mean:",
          sum(1 for x in cmp_rows if x["published_mean"] != "NA"))
    print("abs diff mean  n/median/max:", summ("diff_mean")[:3])
    print("SE ratio       n/median/max/min:", summ("se_ratio", False))
    print("abs diff gap   n/median/max:", summ("diff_gap")[:3])
    print("gap SE ratio   n/median/max/min:", summ("gap_se_ratio", False))
    for x in cmp_rows:
        if any(k in x["note"] for k in ("|diff", "outside", "no published")):
            print(x["CNTRY"], x["diff_mean"], x["se_ratio"], x["diff_gap"], x["gap_se_ratio"], "->", x["note"])


if __name__ == "__main__":
    main()
