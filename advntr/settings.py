import socket


HG19_DIR = '/mnt/hg19_chromosomes/'
CHROMOSOMES = ['chr' + str(chr_number) for chr_number in list(range(1, 23))] + ['chrX', 'chrY']
GENOME_LENGTH = 3100000000
MAX_INSERT_SIZE = 1000

USE_TRAINED_HMMS = False
ILLUMINA_DEFAULT_MODELS_FILE = 'vntr_data/hg19_selected_VNTRs_Illumina.db'
PACBIO_DEFAULT_MODELS_FILE = 'vntr_data/hg19_selected_VNTRs_Pacbio.db'
TRAINED_MODELS_DB = ILLUMINA_DEFAULT_MODELS_FILE
TRAINED_HMMS_DIR = 'vntr_data/'

SCORE_FINDING_READS_FRACTION = 0.0001
SCORE_SELECTION_PERCENTILE = 0
SAVE_SCORE_DISTRIBUTION = False
SCALE_SCORES = True

GC_CONTENT_WINDOW_SIZE = 100
GC_CONTENT_BINS = 10
OUTLIER_COVERAGE = 200

QUALITY_SCORE_CUTOFF = 20
LOW_QUALITY_BP_TO_DISCARD_READ = 0.10
MAPQ_CUTOFF = 0

MAX_ERROR_RATE = 0.05

hostname = socket.gethostname()
if hostname.startswith('genome'):
    CORES = 20
else:
    CORES = 8

FRAMESHIFT_VNTRS = [3056, 25561, 379159, 70186, 188871, 503431, 519759, 301645, 45930]
FRAMESHIFT_VNTRS.append(915594)  # GRCh38, CEL VNTR
LONG_VNTRS = [70186]

USE_ENHANCED_HMM = True
INDEL_MUTATION_MIN_PVALUE = 0.001
INDEL_ERROR_RATE = 0.01
MIN_SUPPORTING_READ_COUNT = 3
USE_ONLY_FULLY_COVERED_RU = False
USE_REF_ALIGNMENT = True

MIN_READ_LENGTH = None

#: Default-off (Task 8, `--prune-reverse`; Tier B, AGENTS.md's two-tier rule). Written
#: from `args.prune_reverse` in advntr/advntr_commands.py the same way `-t/--threads`
#: writes CORES; read (snapshotted before phase 2) in
#: advntr/vntr_finder.py:select_illumina_reads, never inside read_selection.py itself --
#: see that module's docstring for why a per-call bool, not a global read mid-decode,
#: is what keeps phase 2 thread-safe.
PRUNE_REVERSE_DECODE = False


#: Default-off (Task 8, `--exact-frameshift-caller`; Tier B, AGENTS.md's two-tier rule).
#: With it False the decision is the shipped `identify_frameshift`
#: (advntr/vntr_finder.py:187-197) and the code path is byte-for-byte the pre-Task-8 one.
#: With it True the decision is a one-sided exact binomial over Task 7's integer (k, N)
#: and the frozen background at FRAMESHIFT_BACKGROUND_FILE -- see advntr/exact_caller.py.
EXACT_FRAMESHIFT_CALLER = False

#: Path to the operator-supplied frozen background artifact, or None. There is no
#: default and there must not be one: SPEC Q-RATE shows the public candidate-conditioned
#: rates are not plug-in estimates for a production null, so with no artifact the exact
#: caller refuses to run rather than scoring against a number that looks calibrated.
FRAMESHIFT_BACKGROUND_FILE = None
