"""Shared configuration for the ICILS 2023 design-based explainable machine learning study.

All paths are relative to the repository root (the parent of code/):
  RAW   data/spss/ICILS2023_IDB_SPSS/Data   IEA ICILS 2023 International Database, SPSS files
  PROC  data/processed                      analysis file written by 01_prepare.py
  RES   results                             result tables of all analysis steps
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "spss" / "ICILS2023_IDB_SPSS" / "Data"
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
for p in (PROC, RES):
    p.mkdir(parents=True, exist_ok=True)

SEED = 20231101
N_PV = 5
N_REP = 75
PVS = [f"PV{i}CIL" for i in range(1, N_PV + 1)]
REPW = [f"SRWGT{i:02d}" for i in range(1, N_REP + 1)]
WGT = "TOTWGTS"

# --------------------------------------------------------------------------
# Feature dictionary: name -> (block, label, level)
# All ordinal items are recoded in 01_prepare.py so that higher = more.
# --------------------------------------------------------------------------
FEATURES = {
    # Student background
    "S_SEX": ("Background", "Girl (reference: boy)", "student"),
    "S_AGE": ("Background", "Age", "student"),
    "S_IMMBGR": ("Background", "Immigrant background", "student"),
    "S_TLANG": ("Background", "Speaks test language at home", "student"),
    "S_HISEI": ("Background", "Highest parental occupational status (HISEI)", "student"),
    "S_HISCED": ("Background", "Highest parental education (ISCED)", "student"),
    "S_HOMLIT": ("Background", "Books at home", "student"),
    "S_ISCED": ("Background", "Expected educational attainment", "student"),
    # Home ICT
    "S_INTNET": ("Home ICT", "Internet access at home", "student"),
    "H_CONNPROB": ("Home ICT", "Home internet connection problems", "student"),
    "H_NCOMP": ("Home ICT", "Number of computers at home", "student"),
    "H_NTAB": ("Home ICT", "Number of tablets at home", "student"),
    "S_EXCOMP": ("Home ICT", "Years of computer experience", "student"),
    # ICT use
    "U_SCH_WORK": ("ICT use", "ICT use at school for schoolwork", "student"),
    "U_SCH_OTH": ("ICT use", "ICT use at school for other purposes", "student"),
    "U_OUT_WORK": ("ICT use", "ICT use outside school for schoolwork (school days)", "student"),
    "U_OUT_OTH": ("ICT use", "ICT use outside school for other purposes (school days)", "student"),
    "U_WE_WORK": ("ICT use", "ICT use for schoolwork (non-school days)", "student"),
    "U_WE_OTH": ("ICT use", "ICT use for other purposes (non-school days)", "student"),
    "U_LIMIT_SD": ("ICT use", "Parental screen-time limit (school days)", "student"),
    "U_LIMIT_NSD": ("ICT use", "Parental screen-time limit (non-school days)", "student"),
    "A_CHAT": ("ICT use", "Text chat with others", "student"),
    "A_SMPOST": ("ICT use", "Post/view content on social media", "student"),
    "A_SMCHECK": ("ICT use", "Check social media", "student"),
    "A_INFO": ("ICT use", "Use internet to find information", "student"),
    "A_VIDEO": ("ICT use", "Watch online videos/streams", "student"),
    "A_MUSIC": ("ICT use", "Listen to music/radio", "student"),
    # Opportunities to learn information-related internet tasks (focal)
    "OTL_S_FIND": ("OTL info (school)", "Learned at school: find information online", "student"),
    "OTL_S_REFINE": ("OTL info (school)", "Learned at school: refine internet searches", "student"),
    "OTL_S_EVAL": ("OTL info (school)", "Learned at school: evaluate reliability of online information", "student"),
    "OTL_S_REF": ("OTL info (school)", "Learned at school: reference internet sources", "student"),
    "OTL_S_SCAM": ("OTL info (school)", "Learned at school: judge whether a message is a scam", "student"),
    "OTL_S_PRIV": ("OTL info (school)", "Learned at school: manage privacy settings", "student"),
    "OTL_O_FIND": ("OTL info (outside)", "Learned outside school: find information online", "student"),
    "OTL_O_REFINE": ("OTL info (outside)", "Learned outside school: refine internet searches", "student"),
    "OTL_O_EVAL": ("OTL info (outside)", "Learned outside school: evaluate reliability of online information", "student"),
    "OTL_O_REF": ("OTL info (outside)", "Learned outside school: reference internet sources", "student"),
    "OTL_O_SCAM": ("OTL info (outside)", "Learned outside school: judge whether a message is a scam", "student"),
    "OTL_O_PRIV": ("OTL info (outside)", "Learned outside school: manage privacy settings", "student"),
    "S_LRNSAFE": ("School ICT learning", "Learning about safe/responsible ICT use at school", "student"),
    "S_CTCLS": ("School ICT learning", "Learning about computational thinking in class", "student"),
    "S_GENCLASS": ("School ICT learning", "Use of general ICT applications in class", "student"),
    "S_SPECLASS": ("School ICT learning", "Use of specialist ICT applications in class", "student"),
    "S_ICTSTUD": ("School ICT learning", "Studies an ICT subject this year", "student"),
    # Dispositions
    "S_GENEFF": ("Dispositions", "ICT self-efficacy: general applications", "student"),
    "S_SPECEFF": ("Dispositions", "ICT self-efficacy: specialist applications", "student"),
    "S_ACMULT": ("Dispositions", "Academic-media multitasking", "student"),
    "S_ICTPOSG": ("Dispositions", "Positive beliefs about ICT and society", "student"),
    "S_ICTNEG": ("Dispositions", "Negative beliefs about ICT and society", "student"),
    "S_ICTFUT": ("Dispositions", "Expectations for future ICT use", "student"),
    "S_ICTPOSS": ("Dispositions", "Positive perceptions of learning with ICT", "student"),
    # Teachers (aggregated to school)
    "T_INFOEMPH": ("Teachers (school mean)", "Teacher emphasis on credibility evaluation, deceptive practices and fact-checking", "school"),
    "T_OTHEREMPH": ("Teachers (school mean)", "Teacher emphasis on other ICT capabilities", "school"),
    "T_ITE_INFO": ("Teachers (school mean)", "Share of teachers whose initial education covered evaluating source reliability", "school"),
    "T_PD_INFO": ("Teachers (school mean)", "Teacher professional learning on supporting information evaluation and identifying deceptive practices", "school"),
    "T_ICTEFF": ("Teachers (school mean)", "Teacher ICT self-efficacy", "school"),
    "T_VWPOS": ("Teachers (school mean)", "Teacher positive views on ICT in teaching", "school"),
    "T_VWNEG": ("Teachers (school mean)", "Teacher negative views on ICT in teaching", "school"),
    "T_COLICT": ("Teachers (school mean)", "Teacher collaboration on ICT use", "school"),
    "T_RESRC": ("Teachers (school mean)", "Teacher-reported ICT resources", "school"),
    "T_ICTCLASB": ("Teachers (school mean)", "Teacher ICT use for basic classroom activities", "school"),
    "T_ICTCLASA": ("Teachers (school mean)", "Teacher ICT use for advanced classroom activities", "school"),
    "T_USETOOL": ("Teachers (school mean)", "Teacher use of digital learning tools", "school"),
    "T_PROFLRN": ("Teachers (school mean)", "Teacher participation in ICT professional learning", "school"),
    "T_EMPITE": ("Teachers (school mean)", "ICT emphasis in initial teacher education", "school"),
    "T_TEAPIN": ("Teachers (school mean)", "Inquiry-based teaching approach", "school"),
    "T_CODEMP": ("Teachers (school mean)", "Teacher emphasis on CT-related tasks", "school"),
    "T_EXLES": ("Teachers (school mean)", "Teacher experience using ICT in lessons", "school"),
    # School
    "P_DECEPT": ("School", "Principal: importance of protecting students from deceptive internet practices", "school"),
    "P_VWICT": ("School", "Principal views on ICT for educational outcomes", "school"),
    "P_EXPLRN": ("School", "Principal expectations of teacher ICT use", "school"),
    "P_PRIORS": ("School", "Principal priorities for facilitating ICT use", "school"),
    "C_HINRES": ("School", "ICT-coordinator: computer resource hindrances", "school"),
    "C_HINPED": ("School", "ICT-coordinator: pedagogical resource hindrances", "school"),
    "C_ICTRES": ("School", "ICT-coordinator: availability of ICT resources", "school"),
    "C_RATSTD": ("School", "Students per ICT device available for students (log)", "school"),
    "P_NUMSTD_CAT": ("School", "School size (categorised)", "school"),
    "SCH_HISEI": ("School", "School mean HISEI of peers (leave-one-out)", "school"),
    "SCH_HOMLIT": ("School", "School mean books at home of peers (leave-one-out)", "school"),
}
FEAT = list(FEATURES)
BLOCKS = sorted({v[0] for v in FEATURES.values()})
FOCAL = ["OTL_S_EVAL", "OTL_S_SCAM", "OTL_O_EVAL", "OTL_O_SCAM", "T_INFOEMPH",
         "T_ITE_INFO", "T_PD_INFO", "P_DECEPT"]
