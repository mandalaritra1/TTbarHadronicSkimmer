// Lightweight NanoAOD branch-skim for local testing.
//
// Keeps only the collections/fields the TTbarHadronicSkimmer analysis and the
// GloParTv3 recomb study use; drops PFCands/PFNano, SVs, leptons, Photon/Tau,
// the L1 menu and TrigObj (~1.3 GB of a 2 GB v15 BTV-Nano file). It is a true
// structural TTree::CloneTree copy, so the output is still readable by coffea's
// NanoAODSchema. ALL events are kept (only branches are pruned) -> ~6x smaller.
//
// Usage:
//   root -l -b -q 'tools/skim_nano.C("in.root","out.root")'
//
// Re-skimming a directory of downloads (bash):
//   for f in raw/*.root; do
//     root -l -b -q "tools/skim_nano.C(\"$f\",\"light/$(basename $f)\")"
//   done
//
// "branch not present" warnings (e.g. L1PreFiringWeight, btagWeight, Jet_jetId)
// are harmless: those simply don't exist in 2024 NanoAODv15 MC.
void skim_nano(const char* inf, const char* outf) {
  TFile* fin = TFile::Open(inf);
  TTree* t = (TTree*)fin->Get("Events");
  t->SetBranchStatus("*", 0);

  const char* keep[] = {
    // event-level + weights
    "run", "luminosityBlock", "event", "genWeight",
    "Generator_*", "Pileup_*", "fixedGridRho*", "Rho*",
    "LHEScaleWeight", "nLHEScaleWeight", "LHEPdfWeight", "nLHEPdfWeight",
    "LHEWeight_*", "LHE_*", "LHEPart_*", "nLHEPart",
    "PSWeight", "nPSWeight", "L1PreFiringWeight_*", "btagWeight_*",
    "Flag_*", "HLT_*",
    // AK8 + subjets (central to the analysis): keep whole
    "nFatJet", "FatJet_*", "nSubJet", "SubJet_*",
    // AK4 jets: kinematics + ID + b-tagging + flavour only (drop ML taggers)
    "nJet", "Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass", "Jet_jetId",
    "Jet_area", "Jet_rawFactor", "Jet_nConstituents",
    "Jet_hadronFlavour", "Jet_partonFlavour", "Jet_genJetIdx",
    "Jet_btag*", "Jet_*EmEF", "Jet_*HEF", "Jet_muEF",
    "Jet_chMultiplicity", "Jet_neMultiplicity", "Jet_nElectrons", "Jet_nMuons",
    // gen collections (matching / truth studies)
    "nGenPart", "GenPart_*", "nGenJet", "GenJet_*",
    "nGenJetAK8", "GenJetAK8_*", "nSubGenJetAK8", "SubGenJetAK8_*",
    "GenMET_*", "GenVtx_*", "genTtbarId",
  };
  for (auto k : keep) t->SetBranchStatus(k, 1);

  TFile* fout = TFile::Open(outf, "RECREATE", "", fin->GetCompressionSettings());
  TTree* tn = t->CloneTree(-1, "fast");
  tn->Write();
  printf("wrote %lld events, %d active branches\n",
         tn->GetEntries(), (int)tn->GetListOfBranches()->GetEntries());
  fout->Close();
  fin->Close();
}
