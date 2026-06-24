# corrections.py

import numpy as np
import awkward as ak
from coffea.lumi_tools import LumiMask
import correctionlib
from coffea.jetmet_tools import JetResolutionScaleFactor
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory
from coffea.jetmet_tools import (
    CorrectionLibJECStack, CorrectionLibJEC, CorrectionLibJUNC,
    CorrectionLibJER, CorrectionLibJERSF,
)
from coffea.lookup_tools import extractor
import copy
import gzip
import json as _json
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _base_year(iov):
    """Strip Run-3 sub-era suffixes: 2022preEE->2022, 2023postBPix->2023."""
    for suffix in ('preEE', 'postEE', 'preBPix', 'postBPix'):
        if iov.endswith(suffix):
            return iov[:-len(suffix)]
    return iov


# Run-3 v15 sub-eras: JEC/JER come from the JSON-POG correctionlib files (the
# modern recommended path), not legacy txt. Map each IOV -> jsonpog JME subdir.
_JSONPOG_JME_DIR = {
    "2022preEE":    "2022_Summer22",
    "2022postEE":   "2022_Summer22EE",
    "2023preBPix":  "2023_Summer23",
    "2023postBPix": "2023_Summer23BPix",
    "2024":         "2024_Summer24",
    # 2025 analysis uses Summer24 MC (PPD recommendation), so the MC JEC/JER come
    # from the Summer24 correctionlib files. The 2025 prompt-data residuals + veto
    # map live separately under jsonpog/JME/2025_Prompt25/ (for data, when wired).
    "2025":         "2024_Summer24",
}


class _SplitJERSF:
    """JERSF adapter for the JRV2 'split' layout, where the nominal ScaleFactor
    and a symmetric SFUncertainty live in separate corrections (neither has a
    ``systematic`` input). Mirrors coffea's ``CorrectionLibJERSF`` duck-typed
    interface, returning an ``(N, 3)`` array ordered ``[nom, up, down]`` with
    up = SF + unc, down = SF - unc."""

    def __init__(self, sf_correction, unc_correction):
        self._sf = sf_correction
        self._unc = unc_correction
        self._signature = [inp.name for inp in sf_correction.inputs]

    @property
    def signature(self):
        return self._signature

    def getScaleFactor(self, **kwargs):
        args = tuple(kwargs[name] for name in self._signature)
        nom = self._sf.evaluate(*args)
        unc = self._unc.evaluate(*args)
        return ak.concatenate(
            [nom[:, np.newaxis], (nom + unc)[:, np.newaxis], (nom - unc)[:, np.newaxis]],
            axis=1,
        )


def _build_jersf(cset, jer_tag, jet_type):
    """Return a JERSF adapter, handling both JER layouts: JRV1 (one ScaleFactor
    correction with a ``systematic`` input) and JRV2 (split ScaleFactor +
    SFUncertainty). Version-agnostic so a json.gz swap needs no code change."""
    sf = cset[f"{jer_tag}_MC_ScaleFactor_{jet_type}"]
    if any(inp.name == "systematic" for inp in sf.inputs):
        return CorrectionLibJERSF(sf)
    return _SplitJERSF(sf, cset[f"{jer_tag}_MC_SFUncertainty_{jet_type}"])


def _discover_jme_tags(json_path, jet_type):
    """Find the MC JEC and JER tags inside a JSON-POG JME file by pattern, so the
    code is version-agnostic (drop in a newer V4/JRV2 json.gz, no code change).
    Returns (jec_tag, jer_tag) with the trailing ``_MC...`` stripped, as expected
    by ``CorrectionLibJECStack.from_file`` (which re-appends ``_{data_type}_...``)."""
    with gzip.open(json_path, "rt") as f:
        doc = _json.load(f)
    names = [c["name"] for c in doc.get("corrections", [])]
    cnames = [c["name"] for c in doc.get("compound_corrections", [])]
    jec_suffix = f"_MC_L1L2L3Res_{jet_type}"
    jer_suffix = f"_MC_ScaleFactor_{jet_type}"
    jec = next((n[:-len(jec_suffix)] for n in cnames if n.endswith(jec_suffix)), None)
    jer = next((n[:-len(jer_suffix)] for n in names if n.endswith(jer_suffix)), None)
    if jec is None or jer is None:
        raise ValueError(f"Could not find MC JEC/JER tags in {json_path} for {jet_type} "
                         f"(jec={jec}, jer={jer})")
    return jec, jer


def _GetJECUncertainties_jsonpog(FatJets, events, IOV, R="AK8"):
    """Modern correctionlib (JSON-POG) jet correction + JES/JER variations for the
    Run-3 v15 sub-eras. Returns a ``CorrectedJetsFactory`` output with the same
    ``.JES_jes.{up,down}`` / ``.JER.{up,down}`` contract as the legacy txt path."""
    jet_type = f"{R}PFPuppi"
    subdir = _JSONPOG_JME_DIR[IOV]
    fname = "fatJet_jerc.json.gz" if R == "AK8" else "jet_jerc.json.gz"
    json_path = f"{_PROJECT_ROOT}/data/corrections/jsonpog/JME/{subdir}/{fname}"

    jec_tag, jer_tag = _discover_jme_tags(json_path, jet_type)
    cset = correctionlib.CorrectionSet.from_file(json_path)
    # Build adapters by hand (not from_file) so the JES uncertainty field is named
    # "JES_jes" -- matching the field jets.py reads (corrected_jets.JES_jes.up/down).
    jec_stack = CorrectionLibJECStack(
        jec=CorrectionLibJEC(cset.compound[f"{jec_tag}_MC_L1L2L3Res_{jet_type}"]),
        junc=CorrectionLibJUNC([("jes", cset[f"{jec_tag}_MC_Total_{jet_type}"])]),
        jer=CorrectionLibJER(cset[f"{jer_tag}_MC_PtResolution_{jet_type}"]),
        jersf=_build_jersf(cset, jer_tag, jet_type),
    )

    FatJets["pt_raw"] = (1 - FatJets["rawFactor"]) * FatJets["pt"]
    FatJets["mass_raw"] = (1 - FatJets["rawFactor"]) * FatJets["mass"]
    FatJets["jec_rho"] = ak.broadcast_arrays(events.Rho.fixedGridRhoFastjetAll, FatJets.pt)[0]
    if "pt_gen" not in FatJets.fields:
        FatJets["pt_gen"] = ak.values_astype(ak.fill_none(FatJets.matched_gen.pt, 0), np.float32)

    name_map = jec_stack.blank_name_map
    name_map["JetPt"] = "pt"
    name_map["JetMass"] = "mass"
    name_map["JetEta"] = "eta"
    name_map["JetA"] = "area"
    name_map["JetPhi"] = "phi"
    name_map["ptGenJet"] = "pt_gen"
    name_map["ptRaw"] = "pt_raw"
    name_map["massRaw"] = "mass_raw"
    name_map["Rho"] = "jec_rho"

    factory = CorrectedJetsFactory(name_map, jec_stack)
    return factory.build(FatJets)

def GetFlavorEfficiency(Subjet, Flavor, bdisc): # Return "Flavor" efficiency numerator and denominator
    '''
    Subjet --> awkward array object after preselection i.e. SubJetXY
    Flavor --> integer i.e 5, 4, or 0 (b, c, or udsg)
    '''
    # --- Define pT and Eta for Both Candidates' Subjets (for simplicity) --- #
    pT = ak.flatten(Subjet.pt) # pT of subjet in ttbarcand 
    eta = np.abs(ak.flatten(Subjet.eta)) # eta of 1st subjet in ttbarcand 
    flav = np.abs(ak.flatten(Subjet.hadronFlavour)) # either 'normal' or 'anti' quark
    btag = ak.flatten(Subjet.btagCSVV2) 

    #subjet_btagged = (btag > bdisc)
    
    mask_num = (btag > bdisc) & (flav == Flavor)
    mask_den = (flav == Flavor)

    Eff_Num_pT = pT[mask_num]
    Eff_Denom_pT = pT[mask_den]

    Eff_Num_eta = eta[mask_num]
    Eff_Denom_eta = eta[mask_den]
    
    
    '''
    Eff_Num_pT = np.where(subjet_btagged & (flav == Flavor), pT, -1) # if not collecting pT of subjet, then put non exisitent bin, i.e. -1
    Eff_Num_eta = np.where(subjet_btagged & (flav == Flavor), eta, -1) # if not collecting eta of subjet, then put non exisitent bin, i.e. 5

    Eff_Num_pT = ak.flatten(Eff_Num_pT) # extra step needed for numerator to gaurantee proper shape for filling hists
    Eff_Num_eta = ak.flatten(Eff_Num_eta)

    Eff_Denom_pT = np.where(flav == Flavor, pT, -1)
    Eff_Denom_eta = np.where(flav == Flavor, eta, -1)
    '''
    EffStuff = {
        'Num_pT' : Eff_Num_pT,
        'Num_eta' : Eff_Num_eta,
        'Denom_pT' : Eff_Denom_pT,
        'Denom_eta' : Eff_Denom_eta,
    }

    return EffStuff


    
def GetJECUncertainties(FatJets, events, IOV, R='AK8', isData=False):

    # original code https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/jmeCorrections.py

    # Run-3 MC (2022/2023 sub-eras + 2024) uses the modern correctionlib (JSON-POG)
    # path; the legacy txt branches below are kept for Run-2 (and for data, which
    # does not re-apply JEC in this analysis).
    if (not isData) and IOV in _JSONPOG_JME_DIR:
        return _GetJECUncertainties_jsonpog(FatJets, events, IOV, R=R)

    #chspuppi = 'Puppi' if 'AK8' in R else 'chs'
    chspuppi = "Puppi" # always puppi in run3

    jer_tag=None
    if (IOV=='2018'):
        jec_tag="Summer19UL18_V5_MC"
        jec_tag_data={
            "RunA": "Summer19UL18_RunA_V5_DATA",
            "RunB": "Summer19UL18_RunB_V5_DATA",
            "RunC": "Summer19UL18_RunC_V5_DATA",
            "RunD": "Summer19UL18_RunD_V5_DATA",
        }
        jer_tag = "Summer19UL18_JRV2_MC"
    elif (IOV=='2017'):
        jec_tag="Summer19UL17_V5_MC"
        jec_tag_data={
            "RunB": "Summer19UL17_RunB_V5_DATA",
            "RunC": "Summer19UL17_RunC_V5_DATA",
            "RunD": "Summer19UL17_RunD_V5_DATA",
            "RunE": "Summer19UL17_RunE_V5_DATA",
            "RunF": "Summer19UL17_RunF_V5_DATA",
        }
        jer_tag = "Summer19UL17_JRV3_MC"
    elif (IOV=='2016'):
        jec_tag="Summer19UL16_V7_MC"
        jec_tag_data={
            "RunF": "Summer19UL16_RunFGH_V7_DATA",
            "RunG": "Summer19UL16_RunFGH_V7_DATA",
            "RunH": "Summer19UL16_RunFGH_V7_DATA",
        }
        jer_tag = "Summer20UL16_JRV3_MC"
    
    elif (IOV=='2016APV'):
        jec_tag="Summer19UL16_V7_MC"
        ## HIPM/APV     : B_ver1, B_ver2, C, D, E, F
        ## non HIPM/APV : F, G, H

        jec_tag_data={
            "RunB_ver1": "Summer19UL16APV_RunBCD_V7_DATA",
            "RunB_ver2": "Summer19UL16APV_RunBCD_V7_DATA",
            "RunC": "Summer19UL16APV_RunBCD_V7_DATA",
            "RunD": "Summer19UL16APV_RunBCD_V7_DATA",
            "RunE": "Summer19UL16APV_RunEF_V7_DATA",
            "RunF": "Summer19UL16APV_RunEF_V7_DATA",
        }
        jer_tag = "Summer20UL16APV_JRV3_MC"
    elif (IOV=="2024"):
        jec_tag = "Summer24Prompt24_V2_MC"

        jec_tag_data= {}

        jer_tag = "Summer23BPixPrompt23RunD_JRV1_MC"
    else:
        raise ValueError(f"Error: Unknown year \"{IOV}\".")



    ext = extractor()
    if not isData:
        # For MC
        ext.add_weight_sets([
            '* * ' + str(_PROJECT_ROOT) + '/data/corrections/JEC/{0}/{0}_L1FastJet_{1}PF{2}.jec.txt'.format(jec_tag, R, chspuppi),
            '* * ' + str(_PROJECT_ROOT) + '/data/corrections/JEC/{0}/{0}_L2Relative_{1}PF{2}.jec.txt'.format(jec_tag, R, chspuppi),
            '* * ' + str(_PROJECT_ROOT) + '/data/corrections/JEC/{0}/{0}_L3Absolute_{1}PF{2}.jec.txt'.format(jec_tag, R, chspuppi),
            '* * ' + str(_PROJECT_ROOT) + '/data/corrections/JEC/{0}/{0}_UncertaintySources_{1}PF{2}.junc.txt'.format(jec_tag, R, chspuppi),
            '* * ' + str(_PROJECT_ROOT) + '/data/corrections/JEC/{0}/{0}_Uncertainty_{1}PF{2}.junc.txt'.format(jec_tag, R, chspuppi),
        ])
    
        if jer_tag:
            ext.add_weight_sets([
                '* * ' + str(_PROJECT_ROOT) + '/data/corrections/JER/{0}/{0}_PtResolution_{1}PF{2}.jr.txt'.format(jer_tag, R, chspuppi),
                '* * ' + str(_PROJECT_ROOT) + '/data/corrections/JER/{0}/{0}_SF_{1}PF{2}.jersf.txt'.format(jer_tag, R, chspuppi),
            ])
    
    else:
        # For data, make sure we don't duplicate
        tags_done = []
        for run, tag in jec_tag_data.items():
            if tag not in tags_done:
                ext.add_weight_sets([
                    '* * ' + str(_PROJECT_ROOT) + '/data/corrections/JEC/{0}/{0}_L1FastJet_{1}PF{2}.jec.txt'.format(tag, R, chspuppi),
                    '* * ' + str(_PROJECT_ROOT) + '/data/corrections/JEC/{0}/{0}_L2Relative_{1}PF{2}.jec.txt'.format(tag, R, chspuppi),
                    '* * ' + str(_PROJECT_ROOT) + '/data/corrections/JEC/{0}/{0}_L3Absolute_{1}PF{2}.jec.txt'.format(tag, R, chspuppi),
                    '* * ' + str(_PROJECT_ROOT) + '/data/corrections/JEC/{0}/{0}_L2L3Residual_{1}PF{2}.jec.txt'.format(tag, R, chspuppi),
                ])
                tags_done += [tag]

    ext.finalize()
    evaluator = ext.make_evaluator()

    if (not isData):
        jec_names = [
            '{0}_L1FastJet_{1}PF{2}'.format(jec_tag, R, chspuppi),
            '{0}_L2Relative_{1}PF{2}'.format(jec_tag, R, chspuppi),
            '{0}_L3Absolute_{1}PF{2}'.format(jec_tag, R, chspuppi),
            '{0}_Uncertainty_{1}PF{2}'.format(jec_tag, R, chspuppi)]

        if jer_tag: 
            jec_names.extend(['{0}_PtResolution_{1}PF{2}'.format(jer_tag, R, chspuppi),
                              '{0}_SF_{1}PF{2}'.format(jer_tag, R, chspuppi)])

    else:
        jec_names={}
        for run, tag in jec_tag_data.items():
            jec_names[run] = [
                '{0}_L1FastJet_{1}PF{2}'.format(tag, R, chspuppi),
                '{0}_L3Absolute_{1}PF{2}'.format(tag, R, chspuppi),
                '{0}_L2Relative_{1}PF{2}'.format(tag, R, chspuppi),
                '{0}_L2L3Residual_{1}PF{2}'.format(tag, R, chspuppi),]
    #typelist = [t for type(evaluator[name]) for name in jec_names]
    #print("Typelist ", typelist)

    if not isData:
        jec_inputs = {name: evaluator[name] for name in jec_names}
        
        
    else:
        jec_names_data = []
        for era in self.eras:
            jec_names_data += jec_names[f'Run{era}']

        jec_inputs = {name: evaluator[name] for name in jec_names_data}
    #print("jec_inputs: ", jec_inputs)
    jec_stack = JECStack(jec_inputs)

    FatJets['pt_raw'] = (1 - FatJets['rawFactor']) * FatJets['pt']
    FatJets['mass_raw'] = (1 - FatJets['rawFactor']) * FatJets['mass']
    FatJets['jec_rho'] = ak.broadcast_arrays(events.Rho.fixedGridRhoFastjetAll, FatJets.pt)[0]
    if "pt_gen" not in FatJets.fields:
        FatJets['pt_gen'] = ak.values_astype(ak.fill_none(FatJets.matched_gen.pt, 0), np.float32)

    name_map = jec_stack.blank_name_map
    name_map['JetPt'] = 'pt'
    name_map['JetMass'] = 'mass'
    name_map['JetEta'] = 'eta'
    name_map['JetA'] = 'area'
    name_map['JetPhi'] = 'phi'
    name_map['ptGenJet'] = 'pt_gen'
    name_map['ptRaw'] = 'pt_raw'
    name_map['massRaw'] = 'mass_raw'
    name_map['Rho'] = 'jec_rho'



    #events_cache = events.caches[0]
    jet_factory = CorrectedJetsFactory(name_map, jec_stack)
    corrected_jets = jet_factory.build(FatJets)#, lazy_cache=events_cache)

    return corrected_jets


def GetPDFWeights(events):
    
    # hessian pdf weights https://arxiv.org/pdf/1510.03865v1.pdf
    # https://github.com/nsmith-/boostedhiggs/blob/master/boostedhiggs/corrections.py#L60
    
    pdf_nom = np.ones(len(events))

    if "LHEPdfWeight" in events.fields:
        
        pdfUnc = ak.std(events.LHEPdfWeight,axis=1)/ak.mean(events.LHEPdfWeight,axis=1)
        pdfUnc = ak.fill_none(pdfUnc, 0.00)
        
        pdf_up = pdf_nom + pdfUnc
        pdf_down = pdf_nom - pdfUnc
        
        
#         arg = events.LHEPdfWeight[:, 1:-2] - np.ones((len(events), 100))
#         summed = ak.sum(np.square(arg), axis=1)
#         pdf_unc = np.sqrt((1. / 99.) * summed)
        
#         pdf_nom = np.ones(len(events))
#         pdf_up = pdf_nom + pdf_unc
#         pdf_down = np.ones(len(events))

    else:
        
        pdf_up = np.ones(len(events))
        pdf_down = np.ones(len(events))
        

    return [pdf_nom, pdf_up, pdf_down]



def GetPUSF(events, IOV):
    # original code https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/TTbarDileptonProcessor.py#L38
    ## json files from: https://gitlab.cern.ch/cms-nanoAOD/jsonpog-integration/-/tree/master/POG/LUM
    
    # map each IOV to its vendored puWeights.json.gz subdir + correction name
    _pu_subdir = {
        "2022preEE":    "2022_Summer22",
        "2022postEE":   "2022_Summer22EE",
        "2023preBPix":  "2023_Summer23",
        "2023postBPix": "2023_Summer23BPix",
    }
    if IOV.endswith("UL"):
        fname = str(_PROJECT_ROOT)+"/data/corrections/puWeights/{0}_UL/puWeights.json.gz".format(IOV)
    elif IOV in ("2024", "2025"):
        # 2025 MC is Summer24 (PPD); reuse the 2024 PU-weight proxy.
        fname = str(_PROJECT_ROOT)+"/data/corrections/puWeights/2023_Summer23BPix/puWeights.json.gz"
    elif IOV in _pu_subdir:
        fname = str(_PROJECT_ROOT)+"/data/corrections/puWeights/{0}/puWeights.json.gz".format(_pu_subdir[IOV])
    hname = {
        "2016APV": "Collisions16_UltraLegacy_goldenJSON",
        "2016"   : "Collisions16_UltraLegacy_goldenJSON",
        "2017"   : "Collisions17_UltraLegacy_goldenJSON",
        "2018"   : "Collisions18_UltraLegacy_goldenJSON",
        "2024"   : "Collisions2023_369803_370790_eraD_GoldenJson",
        "2025"   : "Collisions2023_369803_370790_eraD_GoldenJson",  # Summer24 MC proxy
        "2022preEE":    "Collisions2022_355100_357900_eraBCD_GoldenJson",
        "2022postEE":   "Collisions2022_359022_362760_eraEFG_GoldenJson",
        "2023preBPix":  "Collisions2023_366403_369802_eraBC_GoldenJson",
        "2023postBPix": "Collisions2023_369803_370790_eraD_GoldenJson",
    }
    evaluator = correctionlib.CorrectionSet.from_file(fname)

    puUp = evaluator[hname[str(IOV)]].evaluate(np.array(events.Pileup.nTrueInt), "up")
    puDown = evaluator[hname[str(IOV)]].evaluate(np.array(events.Pileup.nTrueInt), "down")
    puNom = evaluator[hname[str(IOV)]].evaluate(np.array(events.Pileup.nTrueInt), "nominal")

    return [puNom, puUp, puDown]


def getLumiMask(IOV):

    golden_json_path_2022 = str(_PROJECT_ROOT)+"/data/corrections/goldenJsons/Cert_Collisions2022_355100_362760_Golden.json"
    golden_json_path_2023 = str(_PROJECT_ROOT)+"/data/corrections/goldenJsons/Cert_Collisions2023_366442_370790_Golden.json"
    golden_json_path_2024 = str(_PROJECT_ROOT)+"/data/corrections/goldenJsons/Cert_Collisions2024_378981_386951_Golden.json"
    golden_json_path_2025 = str(_PROJECT_ROOT)+"/data/corrections/goldenJsons/Cert_Collisions2025_391658_398860_Golden.json"
    

    masks = {"2022":LumiMask(golden_json_path_2022),
             "2023":LumiMask(golden_json_path_2023),
             "2024":LumiMask(golden_json_path_2024),
             "2025":LumiMask(golden_json_path_2025),
            }

    # sub-era keys (2022preEE, 2023postBPix, ...) share the full-year golden JSON
    return masks[_base_year(IOV)]


def getMETFilter(IOV, events):
 
    # Reference: https://twiki.cern.ch/twiki/bin/viewauth/CMS/MissingETOptionalFiltersRun2#Run_3_2022_and_2023_data_and_MC
    MET_filters = {'2022': [
                                "goodVertices",
                                "globalSuperTightHalo2016Filter",
                                "EcalDeadCellTriggerPrimitiveFilter",
                                "BadPFMuonFilter",
                                "BadPFMuonDzFilter",
                                "eeBadScFilter",
                                "ecalBadCalibFilter",
                                "hfNoisyHitsFilter"
                               ],
                       '2023'   :["goodVertices",
                                  "globalSuperTightHalo2016Filter",
                                  "EcalDeadCellTriggerPrimitiveFilter",
                                  "BadPFMuonFilter",
                                  "BadPFMuonDzFilter",
                                  "hfNoisyHitsFilter",
                                  "eeBadScFilter",
                                  "ecalBadCalibFilter"],
                       '2024'   :["goodVertices",
                                  "globalSuperTightHalo2016Filter",
                                  "EcalDeadCellTriggerPrimitiveFilter",
                                  "BadPFMuonFilter",
                                  "BadPFMuonDzFilter",
                                  "hfNoisyHitsFilter",
                                  "eeBadScFilter",
                                  "ecalBadCalibFilter"],
                       '2025'   :["goodVertices",
                                  "globalSuperTightHalo2016Filter",
                                  "EcalDeadCellTriggerPrimitiveFilter",
                                  "BadPFMuonFilter",
                                  "BadPFMuonDzFilter",
                                  "hfNoisyHitsFilter",
                                  "eeBadScFilter",
                                  "ecalBadCalibFilter"]}
    
    metfilter = np.ones(len(events), dtype='bool')
    for flag in MET_filters[_base_year(IOV)]:
            metfilter &= np.array(events.Flag[flag])
            
    return metfilter


def pTReweighting(pt0, pt1):
        topcand0_wgt = np.exp(0.0615 - 0.0005*pt0)
        topcand1_wgt = np.exp(0.0615 - 0.0005*pt1)
        ttbar_wgt = np.sqrt(topcand0_wgt*topcand1_wgt) # used for re-weighting ttbar MC
        
        return ttbar_wgt
    
    
def GetQ2weights(events):
# https://gitlab.cern.ch/gagarwal/ttbardileptonic/-/blob/master/corrections.py

    q2Nom = np.ones(len(events))
    q2Up = np.ones(len(events))
    q2Down = np.ones(len(events))
    if ("LHEScaleWeight" in events.fields):
        if ak.all(ak.num(events.LHEScaleWeight, axis=1)==9):
            nom = events.LHEScaleWeight[:,4]
            scales = events.LHEScaleWeight[:,[0,1,3,5,7,8]]
            q2Up = ak.max(scales,axis=1)/nom
            q2Down = ak.min(scales,axis=1)/nom 
        elif ak.all(ak.num(events.LHEScaleWeight, axis=1)==8):
            scales = events.LHEScaleWeight[:,[0,1,3,4,6,7]]
            q2Up = ak.max(scales,axis=1)
            q2Down = ak.min(scales,axis=1)

    return q2Nom, q2Up, q2Down
