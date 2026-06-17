#!/usr/bin/env bash
#
# make_zprime_2024_filelists.sh
#
# Generate one ROOT file list (.txt) per DAS dataset for the 2024 Run-3
# ZPrimeToTT NanoAODv15 signal samples.
#
# For every dataset matching the ZPrimeToTT_Par-M-*-W-* family it:
#   * saves the matched dataset names to metadata/datasets_zprime_2024.txt
#   * writes one .txt per dataset into metadata/filelists/zprime_2024/
#     named  <primary-dataset>__<campaign>.txt
#   * prefixes every /store/...root path with the FNAL xrootd redirector
#   * prints a summary (n datasets, n files per dataset, zero-file warnings)
#
# Prerequisites (run before this script):
#   source /cvmfs/cms.cern.ch/cmsset_default.sh
#   export X509_USER_PROXY=$HOME/x509up_u25128   # or: voms-proxy-init -voms cms -rfc
#
# Usage:
#   ./scripts/make_zprime_2024_filelists.sh
#
set -euo pipefail

# --- configuration ---------------------------------------------------------
DATASET_QUERY='/ZPrimeToTT_Par-M-*-W-*_TuneCP5_13p6TeV_madgraph-pythia8/RunIII2024Summer24NanoAODv15*/NANOAODSIM'
REDIRECTOR='root://cmsxrootd.fnal.gov/'

# Resolve repo root as the parent of this script's directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_LIST="${REPO_ROOT}/metadata/datasets_zprime_2024.txt"
FILELIST_DIR="${REPO_ROOT}/metadata/filelists/zprime_2024"

mkdir -p "$(dirname "${DATASET_LIST}")" "${FILELIST_DIR}"

# --- sanity checks ---------------------------------------------------------
if ! command -v dasgoclient >/dev/null 2>&1; then
  echo "ERROR: dasgoclient not found. Run 'source /cvmfs/cms.cern.ch/cmsset_default.sh' first." >&2
  exit 1
fi

# --- query datasets --------------------------------------------------------
echo ">> Querying DAS for datasets matching:"
echo "   ${DATASET_QUERY}"
dasgoclient --query="dataset=${DATASET_QUERY}" | sort -u > "${DATASET_LIST}"

n_datasets=$(grep -c . "${DATASET_LIST}" || true)
echo ">> Found ${n_datasets} dataset(s); list saved to ${DATASET_LIST}"
if [[ "${n_datasets}" -eq 0 ]]; then
  echo "ERROR: no datasets matched the query." >&2
  exit 1
fi

# --- per-dataset file lists ------------------------------------------------
declare -i n_empty=0
declare -i n_total_files=0

printf '\n%-78s %s\n' "OUTPUT FILE" "NFILES"
printf '%s\n' "--------------------------------------------------------------------------------------"

while IFS= read -r dataset; do
  [[ -z "${dataset}" ]] && continue

  # /Primary/Campaign/Tier  ->  Primary__Campaign
  primary=$(printf '%s' "${dataset}" | cut -d/ -f2)
  campaign=$(printf '%s' "${dataset}" | cut -d/ -f3)
  outfile="${FILELIST_DIR}/${primary}__${campaign}.txt"

  # Fetch the file list for this dataset (don't abort the whole run on one failure).
  files=""
  if ! files=$(dasgoclient --query="file dataset=${dataset}" 2>/tmp/das_err.$$); then
    echo "   !! ERROR querying files for ${dataset}:" >&2
    sed 's/^/      /' /tmp/das_err.$$ >&2 || true
  fi
  rm -f /tmp/das_err.$$

  # Sort+unique, prefix the xrootd redirector, write the list.
  printf '%s\n' "${files}" \
    | grep '^/store/' \
    | sort -u \
    | sed "s#^#${REDIRECTOR}#" > "${outfile}" || true

  nfiles=$(grep -c . "${outfile}" || true)
  n_total_files+=${nfiles}

  printf '%-78s %s\n' "${primary}__${campaign}.txt" "${nfiles}"
  if [[ "${nfiles}" -eq 0 ]]; then
    echo "   !! WARNING: zero files for ${dataset}" >&2
    n_empty+=1
  fi
done < "${DATASET_LIST}"

# --- summary ---------------------------------------------------------------
echo
echo ">> Summary"
echo "   datasets matched   : ${n_datasets}"
echo "   total ROOT files   : ${n_total_files}"
echo "   file lists written : ${FILELIST_DIR}"
echo "   empty datasets     : ${n_empty}"
if [[ "${n_empty}" -gt 0 ]]; then
  echo "   !! ${n_empty} dataset(s) returned zero files - see warnings above." >&2
fi
