import numpy as np
from coffea import util
import glob
import os
import mplhep as hep
hep.style.use("CMS")


# re-exported as functions.lumi
try:
    from .lumi import LUMI_PB as lumi
except ImportError:  # imported as a top-level module (python/ on sys.path)
    from lumi import LUMI_PB as lumi

toptag_sf = 0.9
qcd_xs = 1370.0 #pb From https://cms-gen-dev.cern.ch/xsdb

ttbar_xs = {'700to1000': 833.9 * (0.09210), #pb For ttbar mass from 700 to 1000 # 65.49 in xsdb
            '1000toInf': 833.9 * (0.02474) #pb For ttbar mass from 1000 to Inf # 16.36 in xsdb
           }


zprimeDM_xs = {
    '1000': 1.0,
    '1500': 1.0,
    '2000': 1.0,
    '2500': 1.0,
    '3000': 1.0,
    '3500': 1.0,
    '4000': 1.0,
    '4500': 1.0,
    '5000': 1.0
}


zprime1_xs = {
    '1000': 1.0,
    '1200': 1.0,
    '1400': 1.0,
    '1600': 1.0,
    '1800': 1.0,
    '2000': 1.0,
    '2500': 1.0,
    '3000': 1.0,
    '3500': 1.0,
    '4000': 1.0,
    '4500': 1.0
}

zprime10_xs = {
    "1000": 1.0,
    "1200": 1.0,
    "1400": 1.0,
    "1600": 1.0,
    "1800": 1.0,
    "2000": 1.0,
    "2500": 1.0,
    "3000": 1.0,
    "3500": 1.0,
    "4000": 1.0,
    "4500": 1.0,
    "5000": 1.0,
    "6000": 1.0,
    "7000": 1.0
}

zprime30_xs = {
    "1000": 1.0,
    "1200": 1.0,
    "1400": 1.0,
    "1600": 1.0,
    "1800": 1.0,
    "2000": 1.0,
    "2500": 1.0,
    "3000": 1.0,
    "3500": 1.0,
    "4000": 1.0,
    "4500": 1.0,
    "5000": 1.0,
    "6000": 1.0,
    "7000": 1.0
}

rsgluon_xs = {
    '1000': 1.0,
    '1500': 1.0,
    '2000': 1.0,
    '2500': 1.0,
    '3000': 1.0,
    '3500': 1.0,
    '4000': 1.0,
    '4500': 1.0,
    '5000': 1.0,
    '5500': 1.0,
    '6000': 1.0,
}

xs = {
    'QCD': qcd_xs,
    'TTbar': ttbar_xs,
    'RSGluon': rsgluon_xs,
    'ZPrime1': zprime1_xs,
    'ZPrime10': zprime10_xs,
    'ZPrime30': zprime30_xs,
    'ZPrimeDM': zprimeDM_xs,

}




    

def getLabelMap(coffea_dir='outputs/'):
    
    # coffea_dir = os.path.abspath('./').replace('/plots', '', ).replace('/python', '').replace('/mistag','') + '/outputs/'
    coffea_file = max(glob.glob(coffea_dir+'*.coffea'), key=os.path.getctime)
    label_map = util.load(coffea_file)['analysisCategories']
    
    return label_map
    
    

def getRapidity(p4):
    
    return 0.5 * np.log(( p4.energy + p4.pz ) / ( p4.energy - p4.pz ))



    
    
def loadCoffeaFile(dataset='QCD', year='2016', tag='', bkgest=False):
    
    coffea_dir = os.path.abspath('./').replace('/plots', '', ).replace('/python', '').replace('/mistag','') + '/outputs/'
    
    coffeaFiles = getCoffeaFilenames()
        
    if bkgest: 
        bkgest_str = 'weighted'
    else: 
        bkgest_str = 'unweighted'

    if len(tag) > 0:
        file = coffeaFiles[dataset][bkgest_str][year][tag]
    else:
        file = coffeaFiles[dataset][bkgest_str][year]
        
        
    return util.load(file)   
    
    
    
    
def getHist2(hname, ds, year, sum_axes=['anacat'], integrate_axes={}, tag='', coffea_dir='/srv/outputs/tight_jetid1_ak8pftrigger/scale/'):
    
    ######################################################################################
    # hname = histogram name (example: 'ttbarmass')                                      #
    # ds = dataset name (example: 'JetHT')                                               #
    # year = '2016APV', '2016', '2017', '2018', '2016all', 'all'                         #
    # sum_axes = names of axes to sum over for scikit-hep/hist histogram                 #
    # integrate_axes = range to integrate over axis (example: {'anacat': [0,1,2,3,4,5]}) #
    # tag = string on end of file (ie _bkgest or _test)                                  #
    ######################################################################################    
    
#     coffea_dir = os.path.abspath('./').replace('/plots', '', ).replace('/python', '').replace('/mistag','') + '/outputs/scale/'

    # examples: outputs/scale/TTbar_2016.coffea, outputs/scale/QCD_2016all.coffea
    filename = coffea_dir + ds + '_' + year + tag + '.coffea'

    # print('functions.getHist2 filename', filename)
    
    output = util.load(filename)
    
    
    # integrate and sum over axes
    sum_axes_dict = {ax:sum for ax in sum_axes}

    # get histogram
    histo = output[hname][integrate_axes][sum_axes_dict]
            
            
    return histo
    
            
def makeSaveDirectories(coffea_dir='outputs/'):
    
        
    # plots_dir = os.path.abspath('./').replace('/plots', '', ).replace('/python', '').replace('/mistag','') + '/plots'
    # coffea_dir = os.path.abspath('./').replace('/plots', '', ).replace('/python', '').replace('/mistag','') + '/outputs'
   

    print('coffea dir', coffea_dir)
    plots_dir = coffea_dir.replace('outputs','images').replace('scale','').replace('../','')

    print('plots dir', plots_dir)
    
    directories = [
        
        # output coffea and root files
        # coffea_dir+'/scale',
        # coffea_dir+'/twodalphabet',
        # coffea_dir+'/tight',
        # coffea_dir+'/medium',
        # coffea_dir+'/loose',

        # Closure test
        plots_dir+'/png/closureTest/2016all',
        plots_dir+'/png/closureTest/2016APV',
        plots_dir+'/png/closureTest/2016',
        plots_dir+'/png/closureTest/2017',
        plots_dir+'/png/closureTest/2018',
        plots_dir+'/png/closureTest/all',
        plots_dir+'/pdf/closureTest/2016all',
        plots_dir+'/pdf/closureTest/2016APV',
        plots_dir+'/pdf/closureTest/2016',
        plots_dir+'/pdf/closureTest/2017',
        plots_dir+'/pdf/closureTest/2018',
        plots_dir+'/pdf/closureTest/all',

        # systematics
        plots_dir+'/png/systematics/2016all',
        plots_dir+'/png/systematics/2016APV',
        plots_dir+'/png/systematics/2016',
        plots_dir+'/png/systematics/2017',
        plots_dir+'/png/systematics/2018',
        plots_dir+'/png/systematics/all',
        plots_dir+'/pdf/systematics/2016all',
        plots_dir+'/pdf/systematics/2016APV',
        plots_dir+'/pdf/systematics/2016',
        plots_dir+'/pdf/systematics/2017',
        plots_dir+'/pdf/systematics/2018',
        plots_dir+'/pdf/systematics/all',
        
        # systematics checks
        plots_dir+'/png/systematics_checks/2016all',
        plots_dir+'/png/systematics_checks/2016APV',
        plots_dir+'/png/systematics_checks/2016',
        plots_dir+'/png/systematics_checks/2017',
        plots_dir+'/png/systematics_checks/2018',
        plots_dir+'/png/systematics_checks/all',
        plots_dir+'/pdf/systematics_checks/2016all',
        plots_dir+'/pdf/systematics_checks/2016APV',
        plots_dir+'/pdf/systematics_checks/2016',
        plots_dir+'/pdf/systematics_checks/2017',
        plots_dir+'/pdf/systematics_checks/2018',
        plots_dir+'/pdf/systematics_checks/all',
        
        # ttbarmass
        plots_dir+'/png/ttbarmass/2016all',
        plots_dir+'/png/ttbarmass/2016APV',
        plots_dir+'/png/ttbarmass/2016',
        plots_dir+'/png/ttbarmass/2017',
        plots_dir+'/png/ttbarmass/2018',
        plots_dir+'/png/ttbarmass/all',
        plots_dir+'/pdf/ttbarmass/2016all',
        plots_dir+'/pdf/ttbarmass/2016APV',
        plots_dir+'/pdf/ttbarmass/2016',
        plots_dir+'/pdf/ttbarmass/2017',
        plots_dir+'/pdf/ttbarmass/2018',
        plots_dir+'/pdf/ttbarmass/all',
        
        # kinematics plots
        plots_dir+'/png/kinematics/2016all',
        plots_dir+'/png/kinematics/2016APV',
        plots_dir+'/png/kinematics/2016',
        plots_dir+'/png/kinematics/2017',
        plots_dir+'/png/kinematics/2018',
        plots_dir+'/png/kinematics/all',
        plots_dir+'/pdf/kinematics/2016all',
        plots_dir+'/pdf/kinematics/2016APV',
        plots_dir+'/pdf/kinematics/2016',
        plots_dir+'/pdf/kinematics/2017',
        plots_dir+'/pdf/kinematics/2018',
        plots_dir+'/pdf/kinematics/all',
        
    ]


    for path in directories:
        if not os.path.exists(path):
            os.makedirs(path)



def getCoffeaFilenames():
    
    coffea_dir = os.path.abspath('./').replace('/plots', '', ).replace('/python', '').replace('/mistag','') + '/outputs/'
#     coffea_dir = os.path.abspath('./').replace('/plots', '', ).replace('/python', '').replace('/mistag','') + '/local/history/transferfxn/coffea/'
    
    
    
    coffeaFiles = {
        "JetHT":{
            "unweighted": {
                "2016APV": {
                    "B": coffea_dir+'JetHT_2016APVB.coffea',
                    "C": coffea_dir+'JetHT_2016APVC.coffea',
                    "D": coffea_dir+'JetHT_2016APVD.coffea',
                    "E": coffea_dir+'JetHT_2016APVE.coffea',
                    "F": coffea_dir+'JetHT_2016APVF.coffea'
                },
                "2016": {
                    "F": coffea_dir+'JetHT_2016F.coffea',
                    "G": coffea_dir+'JetHT_2016G.coffea',
                    "H": coffea_dir+'JetHT_2016H.coffea'

                },
               "2017": {
                    "B": coffea_dir+'JetHT_2017B.coffea',
                    "C": coffea_dir+'JetHT_2017C.coffea',
                    "D": coffea_dir+'JetHT_2017D.coffea',
                    "E": coffea_dir+'JetHT_2017E.coffea',
                    "F": coffea_dir+'JetHT_2017F.coffea'
                },
                "2018": {
                    "A": coffea_dir+'JetHT_2018A.coffea',
                    "B": coffea_dir+'JetHT_2018B.coffea',
                    "C": coffea_dir+'JetHT_2018C.coffea',
                    "D": coffea_dir+'JetHT_2018D.coffea'
                }
            },
            "weighted": {
                "2016APV": {
                    "B": coffea_dir+'JetHT_2016APVB_bkgest.coffea',
                    "C": coffea_dir+'JetHT_2016APVC_bkgest.coffea',
                    "D": coffea_dir+'JetHT_2016APVD_bkgest.coffea',
                    "E": coffea_dir+'JetHT_2016APVE_bkgest.coffea',
                    "F": coffea_dir+'JetHT_2016APVF_bkgest.coffea'
                },
                "2016": {
                    "F": coffea_dir+'JetHT_2016F_bkgest.coffea',
                    "G": coffea_dir+'JetHT_2016G_bkgest.coffea',
                    "H": coffea_dir+'JetHT_2016H_bkgest.coffea'

                },
                "2017": {
                    "B": coffea_dir+'JetHT_2017B_bkgest.coffea',
                    "C": coffea_dir+'JetHT_2017C_bkgest.coffea',
                    "D": coffea_dir+'JetHT_2017D_bkgest.coffea',
                    "E": coffea_dir+'JetHT_2017E_bkgest.coffea',
                    "F": coffea_dir+'JetHT_2017F_bkgest.coffea'
                },
                "2018": {
                    "A": coffea_dir+'JetHT_2018A_bkgest.coffea',
                    "B": coffea_dir+'JetHT_2018B_bkgest.coffea',
                    "C": coffea_dir+'JetHT_2018C_bkgest.coffea',
                    "D": coffea_dir+'JetHT_2018D_bkgest.coffea'
                }
            }
        },

        "TTbar": {
            "unweighted": {
                "2016APV": {
                    "700to1000": coffea_dir+'TTbar_2016APV_700to1000.coffea',
                    "1000toInf": coffea_dir+'TTbar_2016APV_1000toInf.coffea'
                },
                "2016": {
                    "700to1000": coffea_dir+'TTbar_2016_700to1000.coffea',
                    "1000toInf": coffea_dir+'TTbar_2016_1000toInf.coffea'
                },
                "2017": {
                    "700to1000": coffea_dir+'TTbar_2017_700to1000.coffea',
                    "1000toInf": coffea_dir+'TTbar_2017_1000toInf.coffea'
                },
                "2018": {
                    "700to1000": coffea_dir+'TTbar_2018_700to1000.coffea',
                    "1000toInf": coffea_dir+'TTbar_2018_1000toInf.coffea'
                }
            },
            "weighted": {
                "2016APV": {
                    "700to1000": coffea_dir+'TTbar_2016APV_700to1000_bkgest.coffea',
                    "1000toInf": coffea_dir+'TTbar_2016APV_1000toInf_bkgest.coffea'
                },
                "2016": {
                    "700to1000": coffea_dir+'TTbar_2016_700to1000_bkgest.coffea',
                    "1000toInf": coffea_dir+'TTbar_2016_1000toInf_bkgest.coffea'
                },
                "2017": {
                    "700to1000": coffea_dir+'TTbar_2017_700to1000_bkgest.coffea',
                    "1000toInf": coffea_dir+'TTbar_2017_1000toInf_bkgest.coffea'
                },
                "2018": {
                    "700to1000": coffea_dir+'TTbar_2018_700to1000_bkgest.coffea',
                    "1000toInf": coffea_dir+'TTbar_2018_1000toInf_bkgest.coffea'
                }
            }
        },
        
        "QCD": {
            "unweighted": {
                "2016APV": coffea_dir+'QCD_2016APV.coffea',
                "2016": coffea_dir+'QCD_2016.coffea',
                "2017": coffea_dir+'QCD_2017.coffea',
                "2018": coffea_dir+'QCD_2018.coffea'
            },
            "weighted": {
                "2016APV": coffea_dir+'QCD_2016APV_bkgest.coffea',
                "2016": coffea_dir+'QCD_2016_bkgest.coffea',
                "2017": coffea_dir+'QCD_2017_bkgest.coffea',
                "2018": coffea_dir+'QCD_2018_bkgest.coffea'
            }
        },
        
        "ZPrimeDM": {
            "unweighted": {
                "2016APV": {
                    "1000": coffea_dir+'ZPrime1000_DM_2016APV.coffea',
                    "1500": coffea_dir+'ZPrime1500_DM_2016APV.coffea',
                    "2000": coffea_dir+'ZPrime2000_DM_2016APV.coffea',
                    "2500": coffea_dir+'ZPrime2500_DM_2016APV.coffea',
                    "3000": coffea_dir+'ZPrime3000_DM_2016APV.coffea',
                    "3500": coffea_dir+'ZPrime3500_DM_2016APV.coffea',
                    "4000": coffea_dir+'ZPrime4000_DM_2016APV.coffea',
                    "4500": coffea_dir+'ZPrime4500_DM_2016APV.coffea',
                    "5000": coffea_dir+'ZPrime5000_DM_2016APV.coffea'
                },
                "2016": {
                    "1000": coffea_dir+'ZPrime1000_DM_2016.coffea',
                    "1500": coffea_dir+'ZPrime1500_DM_2016.coffea',
                    "2000": coffea_dir+'ZPrime2000_DM_2016.coffea',
                    "2500": coffea_dir+'ZPrime2500_DM_2016.coffea',
                    "3000": coffea_dir+'ZPrime3000_DM_2016.coffea',
                    "3500": coffea_dir+'ZPrime3500_DM_2016.coffea',
                    "4000": coffea_dir+'ZPrime4000_DM_2016.coffea',
                    "4500": coffea_dir+'ZPrime4500_DM_2016.coffea',
                    "5000": coffea_dir+'ZPrime5000_DM_2016.coffea'
                },
                "2017": {
                    "1000": coffea_dir+'ZPrime1000_DM_2017.coffea',
                    "1500": coffea_dir+'ZPrime1500_DM_2017.coffea',
                    "2000": coffea_dir+'ZPrime2000_DM_2017.coffea',
                    "2500": coffea_dir+'ZPrime2500_DM_2017.coffea',
                    "3000": coffea_dir+'ZPrime3000_DM_2017.coffea',
                    "3500": coffea_dir+'ZPrime3500_DM_2017.coffea',
                    "4000": coffea_dir+'ZPrime4000_DM_2017.coffea',
                    "4500": coffea_dir+'ZPrime4500_DM_2017.coffea',
                    "5000": coffea_dir+'ZPrime5000_DM_2017.coffea'
                },
                "2018": {
                    "1000": coffea_dir+'ZPrime1000_DM_2018.coffea',
                    "1500": coffea_dir+'ZPrime1500_DM_2018.coffea',
                    "2000": coffea_dir+'ZPrime2000_DM_2018.coffea',
                    "2500": coffea_dir+'ZPrime2500_DM_2018.coffea',
                    "3000": coffea_dir+'ZPrime3000_DM_2018.coffea',
                    "3500": coffea_dir+'ZPrime3500_DM_2018.coffea',
                    "4000": coffea_dir+'ZPrime4000_DM_2018.coffea',
                    "4500": coffea_dir+'ZPrime4500_DM_2018.coffea',
                    "5000": coffea_dir+'ZPrime5000_DM_2018.coffea'
                }
            }
        },
        
        "ZPrime1": {
            "unweighted": {
                "2016APV": {
                    "1000": coffea_dir+'ZPrime1000_1_2016APV.coffea',
                    "1200": coffea_dir+'ZPrime1200_1_2016APV.coffea',
                    "1400": coffea_dir+'ZPrime1400_1_2016APV.coffea',
                    "1600": coffea_dir+'ZPrime1600_1_2016APV.coffea',
                    "1800": coffea_dir+'ZPrime1800_1_2016APV.coffea',
                    "2000": coffea_dir+'ZPrime2000_1_2016APV.coffea',
                    "2500": coffea_dir+'ZPrime2500_1_2016APV.coffea',
                    "3000": coffea_dir+'ZPrime3000_1_2016APV.coffea',
                    "3500": coffea_dir+'ZPrime3500_1_2016APV.coffea',
                    "4000": coffea_dir+'ZPrime4000_1_2016APV.coffea',
                    "4500": coffea_dir+'ZPrime4500_1_2016APV.coffea'
                },
                "2016": {
                    "1000": coffea_dir+'ZPrime1000_1_2016.coffea',
                    "1200": coffea_dir+'ZPrime1200_1_2016.coffea',
                    "1400": coffea_dir+'ZPrime1400_1_2016.coffea',
                    "1600": coffea_dir+'ZPrime1600_1_2016.coffea',
                    "1800": coffea_dir+'ZPrime1800_1_2016.coffea',
                    "2000": coffea_dir+'ZPrime2000_1_2016.coffea',
                    "2500": coffea_dir+'ZPrime2500_1_2016.coffea',
                    "3000": coffea_dir+'ZPrime3000_1_2016.coffea',
                    "3500": coffea_dir+'ZPrime3500_1_2016.coffea',
                    "4000": coffea_dir+'ZPrime4000_1_2016.coffea',
                    "4500": coffea_dir+'ZPrime4500_1_2016.coffea'
                },
                "2017": {
                    "1000": coffea_dir+'ZPrime1000_1_2017.coffea',
                    "1200": coffea_dir+'ZPrime1200_1_2017.coffea',
                    "1400": coffea_dir+'ZPrime1400_1_2017.coffea',
                    "1600": coffea_dir+'ZPrime1600_1_2017.coffea',
                    "1800": coffea_dir+'ZPrime1800_1_2017.coffea',
                    "2000": coffea_dir+'ZPrime2000_1_2017.coffea',
                    "2500": coffea_dir+'ZPrime2500_1_2017.coffea',
                    "3000": coffea_dir+'ZPrime3000_1_2017.coffea',
                    "3500": coffea_dir+'ZPrime3500_1_2017.coffea',
                    "4000": coffea_dir+'ZPrime4000_1_2017.coffea',
                    "4500": coffea_dir+'ZPrime4500_1_2017.coffea'
                },
                "2018": {
                    "1000": coffea_dir+'ZPrime1000_1_2018.coffea',
                    "1200": coffea_dir+'ZPrime1200_1_2018.coffea',
                    "1400": coffea_dir+'ZPrime1400_1_2018.coffea',
                    "1600": coffea_dir+'ZPrime1600_1_2018.coffea',
                    "1800": coffea_dir+'ZPrime1800_1_2018.coffea',
                    "2000": coffea_dir+'ZPrime2000_1_2018.coffea',
                    "2500": coffea_dir+'ZPrime2500_1_2018.coffea',
                    "3000": coffea_dir+'ZPrime3000_1_2018.coffea',
                    "3500": coffea_dir+'ZPrime3500_1_2018.coffea',
                    "4000": coffea_dir+'ZPrime4000_1_2018.coffea',
                    "4500": coffea_dir+'ZPrime4500_1_2018.coffea'
                }
            }
        },
        
        "ZPrime10": {
            "unweighted": {
                "2016APV": {
                    "1000": coffea_dir+'ZPrime1000_10_2016APV.coffea',
                    "1200": coffea_dir+'ZPrime1200_10_2016APV.coffea',
                    "1400": coffea_dir+'ZPrime1400_10_2016APV.coffea',
                    "1600": coffea_dir+'ZPrime1600_10_2016APV.coffea',
                    "1800": coffea_dir+'ZPrime1800_10_2016APV.coffea',
                    "2000": coffea_dir+'ZPrime2000_10_2016APV.coffea',
                    "2500": coffea_dir+'ZPrime2500_10_2016APV.coffea',
                    "3000": coffea_dir+'ZPrime3000_10_2016APV.coffea',
                    "3500": coffea_dir+'ZPrime3500_10_2016APV.coffea',
                    "4000": coffea_dir+'ZPrime4000_10_2016APV.coffea',
                    "4500": coffea_dir+'ZPrime4500_10_2016APV.coffea',
                    "5000": coffea_dir+'ZPrime5000_10_2016APV.coffea',
                    "6000": coffea_dir+'ZPrime6000_10_2016APV.coffea',
                    "7000": coffea_dir+'ZPrime7000_10_2016APV.coffea'
                },
                "2016": {
                    "1000": coffea_dir+'ZPrime1000_10_2016.coffea',
                    "1200": coffea_dir+'ZPrime1200_10_2016.coffea',
                    "1400": coffea_dir+'ZPrime1400_10_2016.coffea',
                    "1600": coffea_dir+'ZPrime1600_10_2016.coffea',
                    "1800": coffea_dir+'ZPrime1800_10_2016.coffea',
                    "2000": coffea_dir+'ZPrime2000_10_2016.coffea',
                    "2500": coffea_dir+'ZPrime2500_10_2016.coffea',
                    "3000": coffea_dir+'ZPrime3000_10_2016.coffea',
                    "3500": coffea_dir+'ZPrime3500_10_2016.coffea',
                    "4000": coffea_dir+'ZPrime4000_10_2016.coffea',
                    "4500": coffea_dir+'ZPrime4500_10_2016.coffea',
                    "5000": coffea_dir+'ZPrime5000_10_2016.coffea',
                    "6000": coffea_dir+'ZPrime6000_10_2016.coffea',
                    "7000": coffea_dir+'ZPrime7000_10_2016.coffea'
                },
                "2017": {
                    "1000": coffea_dir+'ZPrime1000_10_2017.coffea',
                    "1200": coffea_dir+'ZPrime1200_10_2017.coffea',
                    "1400": coffea_dir+'ZPrime1400_10_2017.coffea',
                    "1600": coffea_dir+'ZPrime1600_10_2017.coffea',
                    "1800": coffea_dir+'ZPrime1800_10_2017.coffea',
                    "2000": coffea_dir+'ZPrime2000_10_2017.coffea',
                    "2500": coffea_dir+'ZPrime2500_10_2017.coffea',
                    "3000": coffea_dir+'ZPrime3000_10_2017.coffea',
                    "3500": coffea_dir+'ZPrime3500_10_2017.coffea',
                    "4000": coffea_dir+'ZPrime4000_10_2017.coffea',
                    "4500": coffea_dir+'ZPrime4500_10_2017.coffea',
                    "5000": coffea_dir+'ZPrime5000_10_2017.coffea',
                    "6000": coffea_dir+'ZPrime6000_10_2017.coffea',
                    "7000": coffea_dir+'ZPrime7000_10_2017.coffea'
                },
                "2018": {
                    "1000": coffea_dir+'ZPrime1000_10_2018.coffea',
                    "1200": coffea_dir+'ZPrime1200_10_2018.coffea',
                    "1400": coffea_dir+'ZPrime1400_10_2018.coffea',
                    "1600": coffea_dir+'ZPrime1600_10_2018.coffea',
                    "1800": coffea_dir+'ZPrime1800_10_2018.coffea',
                    "2000": coffea_dir+'ZPrime2000_10_2018.coffea',
                    "2500": coffea_dir+'ZPrime2500_10_2018.coffea',
                    "3000": coffea_dir+'ZPrime3000_10_2018.coffea',
                    "3500": coffea_dir+'ZPrime3500_10_2018.coffea',
                    "4000": coffea_dir+'ZPrime4000_10_2018.coffea',
                    "4500": coffea_dir+'ZPrime4500_10_2018.coffea',
                    "5000": coffea_dir+'ZPrime5000_10_2018.coffea',
                    "6000": coffea_dir+'ZPrime6000_10_2018.coffea',
                    "7000": coffea_dir+'ZPrime7000_10_2018.coffea'
                }
            }
        },
            
        "ZPrime30": {
            "unweighted": {
                "2016APV": {
                    "1000": coffea_dir+'ZPrime1000_30_2016APV.coffea',
                    "1200": coffea_dir+'ZPrime1200_30_2016APV.coffea',
                    "1400": coffea_dir+'ZPrime1400_30_2016APV.coffea',
                    "1600": coffea_dir+'ZPrime1600_30_2016APV.coffea',
                    "1800": coffea_dir+'ZPrime1800_30_2016APV.coffea',
                    "2000": coffea_dir+'ZPrime2000_30_2016APV.coffea',
                    "2500": coffea_dir+'ZPrime2500_30_2016APV.coffea',
                    "3000": coffea_dir+'ZPrime3000_30_2016APV.coffea',
                    "3500": coffea_dir+'ZPrime3500_30_2016APV.coffea',
                    "4000": coffea_dir+'ZPrime4000_30_2016APV.coffea',
                    "4500": coffea_dir+'ZPrime4500_30_2016APV.coffea',
                    "5000": coffea_dir+'ZPrime5000_30_2016APV.coffea',
                    "6000": coffea_dir+'ZPrime6000_30_2016APV.coffea',
                    "7000": coffea_dir+'ZPrime7000_30_2016APV.coffea'
                },
                "2016": {
                    "1000": coffea_dir+'ZPrime1000_30_2016.coffea',
                    "1200": coffea_dir+'ZPrime1200_30_2016.coffea',
                    "1400": coffea_dir+'ZPrime1400_30_2016.coffea',
                    "1600": coffea_dir+'ZPrime1600_30_2016.coffea',
                    "1800": coffea_dir+'ZPrime1800_30_2016.coffea',
                    "2000": coffea_dir+'ZPrime2000_30_2016.coffea',
                    "2500": coffea_dir+'ZPrime2500_30_2016.coffea',
                    "3000": coffea_dir+'ZPrime3000_30_2016.coffea',
                    "3500": coffea_dir+'ZPrime3500_30_2016.coffea',
                    "4000": coffea_dir+'ZPrime4000_30_2016.coffea',
                    "4500": coffea_dir+'ZPrime4500_30_2016.coffea',
                    "5000": coffea_dir+'ZPrime5000_30_2016.coffea',
                    "6000": coffea_dir+'ZPrime6000_30_2016.coffea',
                    "7000": coffea_dir+'ZPrime7000_30_2016.coffea'
                },
                "2017": {
                    "1000": coffea_dir+'ZPrime1000_30_2017.coffea',
                    "1200": coffea_dir+'ZPrime1200_30_2017.coffea',
                    "1400": coffea_dir+'ZPrime1400_30_2017.coffea',
                    "1600": coffea_dir+'ZPrime1600_30_2017.coffea',
                    "1800": coffea_dir+'ZPrime1800_30_2017.coffea',
                    "2000": coffea_dir+'ZPrime2000_30_2017.coffea',
                    "2500": coffea_dir+'ZPrime2500_30_2017.coffea',
                    "3000": coffea_dir+'ZPrime3000_30_2017.coffea',
                    "3500": coffea_dir+'ZPrime3500_30_2017.coffea',
                    "4000": coffea_dir+'ZPrime4000_30_2017.coffea',
                    "4500": coffea_dir+'ZPrime4500_30_2017.coffea',
                    "5000": coffea_dir+'ZPrime5000_30_2017.coffea',
                    "6000": coffea_dir+'ZPrime6000_30_2017.coffea',
                    "7000": coffea_dir+'ZPrime7000_30_2017.coffea'
                },
                "2018": {
                    "1000": coffea_dir+'ZPrime1000_30_2018.coffea',
                    "1200": coffea_dir+'ZPrime1200_30_2018.coffea',
                    "1400": coffea_dir+'ZPrime1400_30_2018.coffea',
                    "1600": coffea_dir+'ZPrime1600_30_2018.coffea',
                    "1800": coffea_dir+'ZPrime1800_30_2018.coffea',
                    "2000": coffea_dir+'ZPrime2000_30_2018.coffea',
                    "2500": coffea_dir+'ZPrime2500_30_2018.coffea',
                    "3000": coffea_dir+'ZPrime3000_30_2018.coffea',
                    "3500": coffea_dir+'ZPrime3500_30_2018.coffea',
                    "4000": coffea_dir+'ZPrime4000_30_2018.coffea',
                    "4500": coffea_dir+'ZPrime4500_30_2018.coffea',
                    "5000": coffea_dir+'ZPrime5000_30_2018.coffea',
                    "6000": coffea_dir+'ZPrime6000_30_2018.coffea',
                    "7000": coffea_dir+'ZPrime7000_30_2018.coffea'
                }
            }
        },
        
        "RSGluon": {
            "unweighted": {
                "2016": {
                    "1000": coffea_dir+'RSGluon1000_2016.coffea',
                    "1500": coffea_dir+'RSGluon1500_2016.coffea',
                    "2000": coffea_dir+'RSGluon2000_2016.coffea',
                    "2500": coffea_dir+'RSGluon2500_2016.coffea',
                    "3000": coffea_dir+'RSGluon3000_2016.coffea',
                    "3500": coffea_dir+'RSGluon3500_2016.coffea',
                    "4000": coffea_dir+'RSGluon4000_2016.coffea',
                    "4500": coffea_dir+'RSGluon4500_2016.coffea',
                    "5000": coffea_dir+'RSGluon5000_2016.coffea',
                    "5500": coffea_dir+'RSGluon5500_2016.coffea',
                    "6000": coffea_dir+'RSGluon6000_2016.coffea'
                },
                "2016APV": {
                    "1000": coffea_dir+'RSGluon1000_2016APV.coffea',
                    "1500": coffea_dir+'RSGluon1500_2016APV.coffea',
                    "2000": coffea_dir+'RSGluon2000_2016APV.coffea',
                    "2500": coffea_dir+'RSGluon2500_2016APV.coffea',
                    "3000": coffea_dir+'RSGluon3000_2016APV.coffea',
                    "3500": coffea_dir+'RSGluon3500_2016APV.coffea',
                    "4000": coffea_dir+'RSGluon4000_2016APV.coffea',
                    "4500": coffea_dir+'RSGluon4500_2016APV.coffea',
                    "5000": coffea_dir+'RSGluon5000_2016APV.coffea',
                    "5500": coffea_dir+'RSGluon5500_2016APV.coffea',
                    "6000": coffea_dir+'RSGluon6000_2016APV.coffea'
                },
                "2017": {
                    "1000": coffea_dir+'RSGluon1000_2017.coffea',
                    "1500": coffea_dir+'RSGluon1500_2017.coffea',
                    "2000": coffea_dir+'RSGluon2000_2017.coffea',
                    "2500": coffea_dir+'RSGluon2500_2017.coffea',
                    "3000": coffea_dir+'RSGluon3000_2017.coffea',
                    "3500": coffea_dir+'RSGluon3500_2017.coffea',
                    "4000": coffea_dir+'RSGluon4000_2017.coffea',
                    "4500": coffea_dir+'RSGluon4500_2017.coffea',
                    "5000": coffea_dir+'RSGluon5000_2017.coffea',
                    "5500": coffea_dir+'RSGluon5500_2017.coffea',
                    "6000": coffea_dir+'RSGluon6000_2017.coffea'
                },
                "2018": {
                    "1000": coffea_dir+'RSGluon1000_2018.coffea',
                    "1500": coffea_dir+'RSGluon1500_2018.coffea',
                    "2000": coffea_dir+'RSGluon2000_2018.coffea',
                    "2500": coffea_dir+'RSGluon2500_2018.coffea',
                    "3000": coffea_dir+'RSGluon3000_2018.coffea',
                    "3500": coffea_dir+'RSGluon3500_2018.coffea',
                    "4000": coffea_dir+'RSGluon4000_2018.coffea',
                    "4500": coffea_dir+'RSGluon4500_2018.coffea',
                    "5000": coffea_dir+'RSGluon5000_2018.coffea',
                    "5500": coffea_dir+'RSGluon5500_2018.coffea',
                    "6000": coffea_dir+'RSGluon6000_2018.coffea'
                }
            }
        }
    }
    
    
        
    return coffeaFiles
    
    
    
    
def printTime(delta, definition='Elapsed Time'):
    
    time_string = '{0:0.0f} h {1:0.0f} min {2:0.0f} sec'.format(delta // 3600, delta % 3600 // 60, delta % 3600 % 60)
    full_string = definition + ': ' + time_string
    print(full_string)
    