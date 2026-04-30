# plotting.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mplhep as hep
hep.style.use("CMS")
from coffea import util
import itertools
import os, sys
import glob
import copy
from hist import intervals
from scipy.optimize import curve_fit
import uproot

sys.path.append('../python/')
import functions

# suppress warnings
import warnings
warnings.filterwarnings("ignore")


# get IOV from command line

if (len(sys.argv) > 1) and (sys.argv[1] in ['2016', '2016APV', '2016all', '2017', '2018', 'all']):
    
    IOV = sys.argv[1]

else:
    
    IOV = '2016all'

    
print(IOV, type(IOV))

# scale factors and luminosity

lumi = functions.lumi
print(lumi)
rsgluon_xs = functions.rsgluon_xs


plotred = 'red'
plotyellow = 'yellow'


# plotred = 'xkcd:deep red'
# plotyellow = 'xkcd:pale gold'




# initialize

savedirstring='ttagSF'
savedir='../outputs/'+savedirstring+'/'


print('making images')
functions.makeSaveDirectories(coffea_dir=savedir)


blind = False
lumifactor = 0.1 if blind else 1.0

label_map = functions.getLabelMap(coffea_dir=savedir)
label_to_int = {label: i for i, label in label_map.items()}

signal_cats = [ i for label, i in label_to_int.items() if '2t' in label]
signal_cen_cats = [ i for label, i in label_to_int.items() if '2t' in label and 'cen' in label]
signal_fwd_cats = [ i for label, i in label_to_int.items() if '2t' in label and 'fwd' in label]

pretag_cats = [ i for label, i in label_to_int.items() if 'pre' in label]
antitag_cats = [ i for label, i in label_to_int.items() if 'at' in label]
antitag_cen_cats = [ i for label, i in label_to_int.items() if 'at' in label and 'cen' in label]
antitag_fwd_cats = [ i for label, i in label_to_int.items() if 'at' in label and 'fwd' in label]


lines_dict = {'solid': 'solid',
 'dotted': (0, (1, 1)),
 'dashed': (0, (5, 5)),
 'dashdot': 'dashdot',
 'loosely dotted': (0, (1, 10)),
 'densely dotted': (0, (1, 1)),
 'long dash with offset': (5, (10, 3)),
 'loosely dashed': (0, (5, 10)),
 'densely dashed': (0, (5, 1)),
 'loosely dashdotted': (0, (3, 10, 1, 10)),
 'dashdotted': (0, (3, 5, 1, 5)),
 'densely dashdotted': (0, (3, 1, 1, 1)),
 'dashdotdotted': (0, (3, 5, 1, 5, 1, 5)),
 'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
 'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))}

lines = list(lines_dict.values())


# categories and systematics labels

# cats = ['0bcen', '0bfwd', '1bcen', '1bfwd', '2bcen', '2bfwd']
# cat_labels = ['cen0b', 'fwd0b', 'cen1b', 'fwd1b', 'cen2b', 'fwd2b']


cats = ['cen', 'fwd']
cat_labels=['cen', 'fwd']


def getUncertainy(hbkg, hUnc):
    
    axes_systematics = hUnc.axes['systematic']
    
    nomvals = hbkg.values()

    for syst in axes_systematics:

        if 'Up' in syst:

            upvals = (1 + np.abs(nomvals - hUnc[{'systematic':syst}].values())/nomvals)

        elif 'Down' in syst:

            downvals = (1 - np.abs(nomvals - hUnc[{'systematic':syst}].values())/nomvals)
            
    return hbkg*upvals, hbkg*downvals




        
# systematics plots #

def plotSystematics(IOV, dataset='TTbar', dirstring=''):

    print('\nPlotting systematics\n')
    
    print(IOV)


    coffea_dir = f'../outputs/{dirstring}/scale/'
    
    

    for cat, catname in zip(['']+cats, ['inclusive']+cat_labels):

        
        
        if catname == 'inclusive':
            
            signal_cat = signal_cats

        elif catname == 'cen':
            
            signal_cat = signal_cen_cats

        elif catname == 'fwd':
            
            signal_cat = signal_fwd_cats
            
        else:
            signal_cat = label_to_int['2t'+cat]
            
        
        
        systsUpDn = list(functions.getHist2('ttbarmass', dataset, IOV, sum_axes=['anacat'], coffea_dir=coffea_dir).axes['systematic'])
        
        systematics = [syst.replace('Up', '') for syst in systsUpDn if 'Down' not in syst]        
        

        for syst in systematics[1:]:

            fig, (ax1, ax2) = plt.subplots(nrows=2, height_ratios=[3, 1])

            text = r'MC TTbar'+'\n'+syst+' systematic variations'

            dytext = ''
            if 'cen' in cat:
                dytext = r'$\bf{central}$'
            elif 'fwd' in cat:
                dytext = r'$\bf{foward}$'

            btext = ''
            if '0b' in cat:
                btext = '0 b-tags'
            elif '1b' in cat:
                btext = '1 b-tag'
            elif '2b' in cat:
                btext = '2 b-tags'

                
            if catname == 'inclusive':
                text = f'MC {dataset}\n{syst} systematic variations\n' + r'b-tag, $\Delta y$ inclusive'
            elif catname == 'cen':
                text = f'MC {dataset}\n{syst} systematic variations\n{dytext}'
            elif catname == 'fwd':
                text = f'MC {dataset}\n{syst} systematic variations\n{dytext}'
            else:
                text = f'MC {dataset}\n{syst} systematic variations\n{btext}, {dytext}'


            hep.cms.label('Preliminary', data=True, lumi='{0:0.1f}'.format(lumi[IOV]*lumifactor/1000.), year=IOV.replace('all',''), loc=2, fontsize=20, ax=ax1)
            hep.cms.text(text, loc=2, fontsize=20, ax=ax1)


            
            if 'hem' in syst:
                
                httbar = functions.getHist2('ttbarmass', dataset, IOV,
                     sum_axes=['anacat'],
                     integrate_axes={'systematic':'nominal', 'anacat':signal_cats},
                     coffea_dir=coffea_dir
                    )
                httbarDn = functions.getHist2('ttbarmass', dataset, IOV,
                     sum_axes=['anacat'],
                     integrate_axes={'systematic':syst, 'anacat':signal_cats},
                     coffea_dir=coffea_dir
                    )


                hep.histplot(httbar, histtype='step', color='k', ax=ax1, label='Nominal')
                hep.histplot(httbarDn, histtype='step', color='red', ax=ax1, label=syst)


                ratioDn = httbarDn / httbar.values()

                hep.histplot(ratioDn, histtype='step', color='red', ax=ax2, label=syst)
                
                
            else:
                httbar = functions.getHist2('ttbarmass', dataset, IOV,
                     sum_axes=['anacat'],
                     integrate_axes={'systematic':'nominal', 'anacat':signal_cats},
                     coffea_dir=coffea_dir
                    )
                httbarUp = functions.getHist2('ttbarmass', dataset, IOV,
                     sum_axes=['anacat'],
                     integrate_axes={'systematic':syst+'Up', 'anacat':signal_cats},
                     coffea_dir=coffea_dir
                    )
                httbarDn = functions.getHist2('ttbarmass', dataset, IOV,
                     sum_axes=['anacat'],
                     integrate_axes={'systematic':syst+'Down', 'anacat':signal_cats},
                     coffea_dir=coffea_dir
                    )

                hep.histplot(httbar, histtype='step', color='k', ax=ax1, label='Nominal')
                hep.histplot(httbarUp, histtype='step', color='green', ax=ax1, label='Up')
                hep.histplot(httbarDn, histtype='step', color='red', ax=ax1, label='Down')


                ratioUp = httbarUp / httbar.values()
                ratioDn = httbarDn / httbar.values()

                hep.histplot(ratioUp, histtype='step', color='green', ax=ax2)
                hep.histplot(ratioDn, histtype='step', color='red', ax=ax2)
            ax2.axhline(1, color='black', ls='--')
            
            ymax = np.max(httbar.values()) * 1.5

            ax2.set_ylabel('Syst/Nom')
            ax2.set_xlabel(ax1.get_xlabel())
            ax1.set_xlabel('')
            ax1.set_ylim(1e-1,ymax)
            ax2.set_ylim(0.5,1.5)


            ax1.legend()

            imagefile = f'images/{dirstring}/png/systematics/{IOV}/{dataset}_{catname}_{syst}.png'

            plt.savefig(imagefile)
            plt.savefig(imagefile.replace('png','pdf'))
            print('saving ', imagefile)
            print('saving ', imagefile.replace('png','pdf'))

            ax1.plot()



                
def plotDataMCRun3(IOV='2023', dirstring='dy'):
    """Data/MC comparison in the antitag (control) region for Run-3 IOVs."""

    print(f'\nPlotting Run-3 Data/MC comparison  IOV={IOV}\n')

    hists = ['jetmsd', 'jetmsd1', 'ttbarmass', 'ht', 'jetdy']

    regions = {
        'acen': antitag_cen_cats,
        'afwd': antitag_fwd_cats,
    }

    coffea_dir = f'../outputs/{dirstring}/scale/'
    fttbar = util.load(coffea_dir + f'TTbar_{IOV}.coffea')
    fdata  = util.load(coffea_dir + f'JetMET_{IOV}.coffea')

    lumitext = '{0:0.1f}'.format(functions.lumi[IOV] / 1000.)

    os.makedirs(f'images/{dirstring}/png/datamc/{IOV}/', exist_ok=True)
    os.makedirs(f'images/{dirstring}/pdf/datamc/{IOV}/', exist_ok=True)

    for catname, cat_list in regions.items():

        for histname in hists:

            integrate_axes = {'systematic': 'nominal', 'anacat': cat_list}

            h_ttbar = fttbar[histname][integrate_axes][{'anacat': sum}]
            h_data  = fdata[histname][integrate_axes][{'anacat': sum}]
            h_bkg   = h_ttbar

            h_unc = fttbar[histname][{'anacat': cat_list}][{'anacat': sum}]
            systUp, systDn = getUncertainy(h_ttbar, h_unc)

            fig, (ax1, ax2) = plt.subplots(nrows=2, height_ratios=[3, 1])

            hep.cms.label('Preliminary', data=True,
                          lumi=lumitext, year=IOV,
                          loc=2, fontsize=20, ax=ax1)

            hep.histplot(h_bkg,  histtype='fill',      color=plotred,   label='SM TTbar', ax=ax1)
            hep.histplot(h_data, histtype='errorbar',  color='black',   label='Data',     ax=ax1)

            ax1.fill_between(h_bkg.axes[0].centers,
                             h_bkg.values() - systDn,
                             h_bkg.values() + systUp,
                             alpha=0.3, color='grey', label='Syst. unc.')

            ax1.legend(fontsize=14)
            ax1.set_xlabel('')
            ax1.set_ylabel('Events')

            ratio_plot = h_data / h_bkg.values()
            hep.histplot(ratio_plot, ax=ax2, histtype='errorbar', color='black')
            ax2.fill_between(h_bkg.axes[0].centers,
                             1. - systDn / h_bkg.values(),
                             1. + systUp / h_bkg.values(),
                             alpha=0.3, color='grey')
            ax2.axhline(1, color='black', ls='--', lw=1)
            ax2.set_ylim(0.5, 1.5)
            ax2.set_ylabel('Data/MC')
            ax2.set_xlabel(h_bkg.axes[0].label)

            plt.tight_layout()

            pngfile = f'images/{dirstring}/png/datamc/{IOV}/{catname}_{histname}.png'
            pdffile = f'images/{dirstring}/pdf/datamc/{IOV}/{catname}_{histname}.pdf'
            plt.savefig(pngfile)
            plt.savefig(pdffile)
            print('saving', pngfile)

            plt.close()


def plotClosureTest():

    
    print('\nPlotting Closure Test\n')
    

    fig, (ax1, ax2) = plt.subplots(nrows=2, height_ratios=[3, 1])



    hbkg = functions.getHist2('ttbarmass', 'JetHT', IOV,
             sum_axes=['anacat'],
             integrate_axes={'systematic':'nominal', 'anacat':antitag_cats},
             tag = '_bkgest'
            )

    httbar = functions.getHist2('ttbarmass', 'TTbar', IOV,
             sum_axes=['anacat'],
             integrate_axes={'systematic':'nominal', 'anacat':signal_cats},        
            )

    hsig = functions.getHist2('ttbarmass', 'JetHT', IOV,
             sum_axes=['anacat'],
             integrate_axes={'systematic':'nominal', 'anacat':signal_cats}        
            )


    # 2D histogram with all uncertainties for getting background uncertainty
    hUnc = functions.getHist2('ttbarmass', 'JetHT', IOV,
             sum_axes=['anacat'],
             integrate_axes={'anacat':antitag_cats},
             tag = '_bkgest')


    hUp, hDn = getUncertainy(hbkg, hUnc)


    text = 'Preliminary'+'\n'+r'$\Delta y$ inclusive'+', '+r'b-tag inclusive'

    hep.cms.label('', data=True, lumi='{0:0.1f}'.format(functions.lumi[IOV]*lumifactor/1000), year=IOV.replace('all',''), loc=2, fontsize=20, ax=ax1)
    hep.cms.text(text, loc=2, fontsize=20, ax=ax1)

    hep.histplot(hsig, histtype='errorbar', color='black', label='Data', ax=ax1)
    hep.histplot(hbkg+httbar, histtype='fill', color=plotyellow, label='NTMJ Bkg Est', ax=ax1)
    hep.histplot(httbar, histtype='fill', color=plotred, label='SM TTbar', ax=ax1)

    height = hUp.values() + hDn.values()
    bottom = hbkg.values() - hDn.values()
    edges  = hbkg.axes['ttbarmass'].edges


    ax1.bar(x = edges[:-1],
               height=height,
               bottom=bottom,
               width = np.diff(edges), align='edge', hatch='//////', edgecolor='gray',
               linewidth=0, facecolor='none', alpha=0.7,
               zorder=10, label='Stat + Syst')


    ratio_plot =  hsig / hbkg.values()
    ratioUp = hUp.values() / hbkg.values()
    ratioDn = hUp.values() / hbkg.values()


    ax2.bar(x = edges[:-1],
               height=(np.ones_like(ratio_plot.values()) + ratioUp + ratioDn),
               bottom=(np.ones_like(ratio_plot.values()) - ratioDn),
               width = np.diff(edges), align='edge', edgecolor='gray',
               linewidth=0, facecolor='gray', alpha=0.3,
               zorder=10, label='Stat + Syst')


    ratio_plot =  hsig / hbkg.values()
    hep.histplot(ratio_plot, ax=ax2, histtype='errorbar', color='black')
    ax2.set_ylim(-10,10)
    ax2.axhline(1, color='black', ls='--')
    ax2.set_ylabel('Data/Bkg')


    ax1.legend()
    ax1.set_ylabel(f'Events / Bin GeV'.replace('j',''))
    ax1.set_yscale('log')
    ax1.set_ylim(1e-1, 1e6)
    ax1.set_xlim(900,6000)
    ax2.set_xlim(900,6000)
    ax1.set_xlabel('')


    savefigname = f'images/{dirstring}/png/closureTest/{IOV}/closure_inclusive.png'
    plt.savefig(savefigname)
    plt.savefig(savefigname.replace('png', 'pdf'))

    print('saving '+savefigname)
    print('saving '+savefigname.replace('png', 'pdf'))






    # plot regions

    for cat in cats:


        fig, (ax1, ax2) = plt.subplots(nrows=2, height_ratios=[3, 1])


        signal_cat = label_to_int['2t'+cat]
        antitag_cat = label_to_int['at'+cat]


        hbkg = functions.getHist2('ttbarmass', 'JetHT', IOV,
             sum_axes=[],
             integrate_axes={'systematic':'nominal', 'anacat':antitag_cat},
             tag = '_bkgest'

                             )

        httbar = functions.getHist2('ttbarmass', 'TTbar', IOV,
                 sum_axes=[],
                 integrate_axes={'systematic':'nominal', 'anacat':signal_cat},        
                )

        hsig = functions.getHist2('ttbarmass', 'JetHT', IOV,
                 sum_axes=[],
                 integrate_axes={'systematic':'nominal', 'anacat':signal_cat}        
                )
        

        # 2D histogram with all uncertainties for getting background uncertainty
        hUnc = functions.getHist2('ttbarmass', 'JetHT', IOV,
                 sum_axes=[],
                 integrate_axes={'anacat':antitag_cat},
                 tag = '_bkgest')


        hUp, hDn = getUncertainy(hbkg, hUnc)


        dytext = ''
        if 'cen' in cat:
            dytext = r'$\Delta y$ < 1.0'
        elif 'fwd' in cat:
            dytext = r'$\Delta y$ > 1.0'

        btext = ''
        if '0b' in cat:
            btext = '0 b-tags'
        elif '1b' in cat:
            btext = '1 b-tag'
        elif '2b' in cat:
            btext = '2 b-tags'

    
        text = 'Preliminary'+'\n'+dytext+', '+ btext
        hep.cms.label('', data=True, lumi='{0:0.1f}'.format(functions.lumi[IOV]*lumifactor/1000), year=IOV.replace('all',''), loc=2, fontsize=20, ax=ax1)
        hep.cms.text(text, loc=2, fontsize=20, ax=ax1)

        hep.histplot(hsig, histtype='errorbar', color='black', label='Data', ax=ax1)
        hep.histplot(hbkg+httbar, histtype='fill', color=plotyellow, label='NTMJ Bkg Est', ax=ax1)
        hep.histplot(httbar, histtype='fill', color=plotred, label='SM TTbar', ax=ax1)

        height = hUp.values() + hDn.values()
        bottom = hbkg.values() - hDn.values()
        edges  = hbkg.axes['ttbarmass'].edges


        ax1.bar(x = edges[:-1],
                   height=height,
                   bottom=bottom,
                   width = np.diff(edges), align='edge', hatch='//////', edgecolor='gray',
                   linewidth=0, facecolor='none', alpha=0.7,
                   zorder=10, label='Stat + Syst')


        ratio_plot =  hsig / hbkg.values()
        ratioUp = hUp.values() / hbkg.values()
        ratioDn = hUp.values() / hbkg.values()


        ax2.bar(x = edges[:-1],
                   height=(np.ones_like(ratio_plot.values()) + ratioUp + ratioDn),
                   bottom=(np.ones_like(ratio_plot.values()) - ratioDn),
                   width = np.diff(edges), align='edge', edgecolor='gray',
                   linewidth=0, facecolor='gray', alpha=0.3,
                   zorder=10, label='Stat + Syst')


        ratio_plot =  hsig / hbkg.values()
        hep.histplot(ratio_plot, ax=ax2, histtype='errorbar', color='black')
        ax2.set_ylim(-10,10)
        ax2.axhline(1, color='black', ls='--')
        ax2.set_ylabel('Data/Bkg')


        ax1.legend()
        ax1.set_ylabel(f'Events / Bin GeV'.replace('j',''))
        ax1.set_yscale('log')
        ax1.set_ylim(1e-1, 1e6)
        ax1.set_xlim(900,6000)
        ax2.set_xlim(900,6000)
        ax1.set_xlabel('')

        savefigname = f'images/{dirstring}/png/closureTest/{IOV}/closure_{cat}.png'
        plt.savefig(savefigname)
        plt.savefig(savefigname.replace('png', 'pdf'))

        print('saving '+savefigname)
        print('saving '+savefigname.replace('png', 'pdf'))


    
    

def plotClosureTestQCD():
    
    
    print('\nPlotting QCD Closure Test\n')


    fig, (ax1, ax2) = plt.subplots(nrows=2, height_ratios=[3, 1])

    hbkg = functions.getHist2('ttbarmass', 'QCD', IOV,
             sum_axes=['anacat'],
             integrate_axes={'systematic':'nominal', 'anacat':antitag_cats},
             tag = '_bkgest'

            )
    
    hsig = functions.getHist2('ttbarmass', 'QCD', IOV,
             sum_axes=['anacat'],
             integrate_axes={'systematic':'nominal', 'anacat':signal_cats},
             tag = ''
            )

    # 2D histogram with all uncertainties for getting background uncertainty
    hUnc = functions.getHist2('ttbarmass', 'QCD', IOV,
             sum_axes=['anacat'],
             integrate_axes={'anacat':antitag_cats},
             tag = '_bkgest')


    hUp, hDn = getUncertainy(hbkg, hUnc)


    text = 'Preliminary'+'\n'+r'$\Delta y$ inclusive'+', '+r'b-tag inclusive'

    hep.cms.label('', data=True, lumi='{0:0.1f}'.format(functions.lumi[IOV]*lumifactor/1000), year=IOV.replace('all','').replace('all',''), loc=2, fontsize=20, ax=ax1)
    hep.cms.text(text, loc=2, fontsize=20, ax=ax1)

    hep.histplot(hsig, histtype='errorbar', color='black', label='QCD SR', ax=ax1)
    hep.histplot(hbkg, histtype='fill', color=plotyellow, label='QCD Bkg Est', ax=ax1)
    herr = np.sqrt(hbkg.variances())

    height = hUp.values() + hDn.values()
    bottom = hbkg.values() - hDn.values()
    edges  = hbkg.axes['ttbarmass'].edges


    ax1.bar(x = edges[:-1],
               height=height,
               bottom=bottom,
               width = np.diff(edges), align='edge', hatch='//////', edgecolor='gray',
               linewidth=0, facecolor='none', alpha=0.7,
               zorder=10, label='Stat + Syst')


    ratio_plot =  hsig / hbkg.values()
    ratioUp = hUp.values() / hbkg.values()
    ratioDn = hUp.values() / hbkg.values()


    ax2.bar(x = edges[:-1],
               height=(np.ones_like(ratio_plot.values()) + ratioUp + ratioDn),
               bottom=(np.ones_like(ratio_plot.values()) - ratioDn),
               width = np.diff(edges), align='edge', edgecolor='gray',
               linewidth=0, facecolor='gray', alpha=0.3,
               zorder=10, label='Stat + Syst')

    hep.histplot(ratio_plot, ax=ax2, histtype='errorbar', color='black')
    ax2.set_ylim(-10,10)
    ax2.axhline(1, color='black', ls='--')
    ax2.set_ylabel('Data/Bkg')


    ax1.legend()
    ax1.set_ylabel(f'Events / Bin GeV'.replace('j',''))
    ax1.set_yscale('log')
    ax1.set_ylim(1e-1, 1e13)
    ax1.set_xlim(900,6000)
    ax2.set_xlim(900,6000)
    ax1.set_xlabel('')

    savefigname = f'images/{dirstring}/png/closureTest/{IOV}/closureQCD_inclusive.png'
    plt.savefig(savefigname)
    plt.savefig(savefigname.replace('png', 'pdf'))

    print('saving '+savefigname)
    print('saving '+savefigname.replace('png', 'pdf'))


    
    # plot regions
    
    for cat in cats:
        
        
        signal_cat = label_to_int['2t'+cat]
        antitag_cat = label_to_int['at'+cat]        

        fig, (ax1, ax2) = plt.subplots(nrows=2, height_ratios=[3, 1])

        hbkg = functions.getHist2('ttbarmass', 'QCD', IOV,
                 sum_axes=[],
                 integrate_axes={'systematic':'nominal', 'anacat':antitag_cat},
                 tag = '_bkgest'

                )

        hsig = functions.getHist2('ttbarmass', 'QCD', IOV,
                 sum_axes=[],
                 integrate_axes={'systematic':'nominal', 'anacat':signal_cat}        
                )


        
        # 2D histogram with all uncertainties for getting background uncertainty
        hUnc = functions.getHist2('ttbarmass', 'QCD', IOV,
                 sum_axes=[],
                 integrate_axes={'anacat':antitag_cat},
                 tag = '_bkgest')


        hUp, hDn = getUncertainy(hbkg, hUnc)



        dytext = ''
        if 'cen' in cat:
            dytext = r'$\Delta y$ < 1.0'
        elif 'fwd' in cat:
            dytext = r'$\Delta y$ > 1.0'

        btext = ''
        if '0b' in cat:
            btext = '0 b-tags'
        elif '1b' in cat:
            btext = '1 b-tag'
        elif '2b' in cat:
            btext = '2 b-tags'

    
        text = 'Preliminary'+'\n'+dytext+', '+ btext

        lumitext = '{0:0.0f}'.format(functions.lumi[IOV]*lumifactor/1000) if IOV=='all' else '{0:0.1f}'.format(functions.lumi[IOV]*lumifactor/1000)
        yeartext = '' if IOV=='all' else IOV.replace('all','')

        hep.cms.label('', data=True, lumi=lumitext, year=yeartext, loc=2, fontsize=20, ax=ax1)
        hep.cms.text(text, loc=2, fontsize=20, ax=ax1)

        hep.histplot(hsig, histtype='errorbar', color='black', label='QCD SR', ax=ax1)
        hep.histplot(hbkg, histtype='fill', color=plotyellow, label='QCD Bkg Est', ax=ax1)
        herr = np.sqrt(hbkg.variances())

        height = hUp.values() + hDn.values()
        bottom = hbkg.values() - hDn.values()
        edges  = hbkg.axes['ttbarmass'].edges


        ax1.bar(x = edges[:-1],
                   height=height,
                   bottom=bottom,
                   width = np.diff(edges), align='edge', hatch='//////', edgecolor='gray',
                   linewidth=0, facecolor='none', alpha=0.7,
                   zorder=10, label='Stat + Syst')
        

        ratio_plot =  hsig / hbkg.values()
        ratioUp = hUp.values() / hbkg.values()
        ratioDn = hUp.values() / hbkg.values()


        ax2.bar(x = edges[:-1],
                   height=(np.ones_like(ratio_plot.values()) + ratioUp + ratioDn),
                   bottom=(np.ones_like(ratio_plot.values()) - ratioDn),
                   width = np.diff(edges), align='edge', edgecolor='gray',
                   linewidth=0, facecolor='gray', alpha=0.3,
                   zorder=10, label='Stat + Syst')

        hep.histplot(ratio_plot, ax=ax2, histtype='errorbar', color='black')
        ax2.set_ylim(-10,10)
        ax2.axhline(1, color='black', ls='--')
        ax2.set_ylabel('Data/Bkg')


        ax1.legend()
        ax1.set_ylabel(f'Events / Bin GeV'.replace('j',''))
        ax1.set_yscale('log')
        ax1.set_ylim(1e-1, 1e13)
        ax1.set_xlim(900,6000)
        ax2.set_xlim(900,6000)
        ax1.set_xlabel('')


        savefigname = f'images/{dirstring}/png/closureTest/{IOV}/closureQCD_{cat}.png'
        plt.savefig(savefigname)
        plt.savefig(savefigname.replace('png', 'pdf'))

        print('saving '+savefigname)
        print('saving '+savefigname.replace('png', 'pdf'))




        
def plotKinematics(IOV, dataset='TTbar', dirstring=''):

    
    print('\nPlotting Kinematics \n')
    

    hists = [
        # 'jetdy',
        # 'jetpt',
        # 'jetpt1',
        # 'jety',
        # 'jety1',
        # 'jeteta',
        # 'jetphi',
        # 'jeteta1',
        # 'jetphi1',
        # 'ttbarmass',
        # 'jetmass',
        'jetmsd',
        # 'jetmass1',
        'jetmsd1',
        # 'ht'
    ]


    regions = [
        'inclusive',
        'cenPass',
        'cenFail',
        'fwdPass',
        'fwdFail'
    ]

    lumitext = '{0:0.0f}'.format(functions.lumi[IOV]*lumifactor/1000) if IOV=='all' else '{0:0.1f}'.format(functions.lumi[IOV]*lumifactor/1000)
    yeartext = '' if IOV=='all' else IOV.replace('all','')

    for catname in regions:

        for histname in hists:
            
            coffea_dir = f'../outputs/{dirstring}/'
            
            fttbar = util.load(coffea_dir+'/scale/TTbar_'+IOV+'.coffea')
            fqcd = util.load(coffea_dir+'/scale/QCD_'+IOV+'.coffea')
            fdata = util.load(coffea_dir+'/scale/JetHT_'+IOV+'.coffea')
            fsig = util.load(coffea_dir+'/scale/RSGluon2000_'+IOV+'.coffea')
    
    
            integrate_axes = {'systematic':'nominal'}

            if IOV=='all': 
                integrate_axes = {}

                if 'cen' in catname:
                    if 'Pass' in catname:
                        integrate_axes = {'anacat':signal_cen_cats}
                    elif 'Fail' in catname:
                        integrate_axes = {'anacat':antitag_cen_cats}
                elif 'fwd' in catname:
                    if 'Pass' in catname:
                        integrate_axes = {'anacat':signal_fwd_cats}
                    elif 'Fail' in catname:
                        integrate_axes = {'anacat':antitag_fwd_cats}


            else:
                if 'cen' in catname:
                    if 'Pass' in catname:
                        integrate_axes = {'systematic':'nominal', 'anacat':signal_cen_cats}
                    elif 'Fail' in catname:
                        integrate_axes = {'systematic':'nominal', 'anacat':antitag_cen_cats}
                elif 'fwd' in catname:
                    if 'Pass' in catname:
                        integrate_axes = {'systematic':'nominal', 'anacat':signal_fwd_cats}
                    elif 'Fail' in catname:
                        integrate_axes = {'systematic':'nominal', 'anacat':antitag_fwd_cats}
        
        

            
            
            h_ttbar = fttbar[histname][integrate_axes][{'anacat':sum}]

            h_qcd = fqcd[histname][integrate_axes][{'anacat':sum}]
            
            h_data = fdata[histname][integrate_axes][{'anacat':sum}]
            
            h_bkg = (h_qcd + h_ttbar)
            
            h_sig = fsig[histname][integrate_axes][{'anacat':sum}]



            
        
        
        


            if 'ttbarmass' in histname or 'jetmass' in histname:
        
                h_bkg = h_bkg[::3j]
                h_ttbar = h_ttbar[::3j]
                h_data = h_data[::3j]
                h_sig = h_sig[::3j]
                h_qcd = h_qcd[::3j]



        
        
            nomvals = h_ttbar.values()
            systUp2 = np.zeros_like(nomvals)
            systDn2 = np.zeros_like(nomvals)



            if IOV == 'all': 

                for yy in ['2016all', '2017', '2018']:
                    fttbar = util.load(coffea_dir+'/scale/TTbar_'+yy+'.coffea')
                    fqcd = util.load(coffea_dir+'/scale/QCD_'+yy+'.coffea')

                    integrate_axes['systematic'] = 'nominal'

                    if 'ttbarmass' in histname or 'jetmass' in histname:
                        nomvals = fttbar[histname][integrate_axes][{'anacat':sum}][::3j].values()
                        qcdnomvals = fqcd[histname][integrate_axes][{'anacat':sum}][::3j].values()
                    
                    else:
                        nomvals = fttbar[histname][integrate_axes][{'anacat':sum}].values()
                        qcdnomvals = fqcd[histname][integrate_axes][{'anacat':sum}].values()




                    for syst in fttbar[histname].axes['systematic']:
        
                        integrate_axes['systematic'] = syst

                        if 'ttbarmass' in histname or 'jetmass' in histname:
                            httbar_syst = fttbar[histname][integrate_axes][{'anacat':sum}][::3j]
                            hqcd_syst = fqcd[histname][integrate_axes][{'anacat':sum}][::3j]
                        else:
                            httbar_syst = fttbar[histname][integrate_axes][{'anacat':sum}]
                            hqcd_syst = fqcd[histname][integrate_axes][{'anacat':sum}]
                    
                        if 'Up' in syst:
                            systvals = np.where(httbar_syst.values()[0] > 0, httbar_syst.values() - nomvals, 0) 
                            systUp2 = systUp2 + systvals * systvals
                        if 'Down' in syst:
                            systvals = np.where(httbar_syst.values()[0] > 0, httbar_syst.values() - nomvals, 0) 
                            systDn2 = systDn2 + systvals * systvals
                    
                    for syst in fqcd[histname].axes['systematic']:
                    
                        if 'Up' in syst:
                            systvals = np.where(hqcd_syst.values()[0] > 0, hqcd_syst.values() - qcdnomvals, 0) 
                            systUp2 = systUp2 + systvals * systvals
                        if 'Down' in syst:
                            systvals = np.where(hqcd_syst.values()[0] > 0, hqcd_syst.values() - qcdnomvals, 0) 
                            systDn2 = systDn2 + systvals * systvals


            else:
                for syst in fttbar[histname].axes['systematic']:
    
                    integrate_axes['systematic'] = syst
                    
                    nomvals = h_ttbar.values()
    
                    if 'ttbarmass' in histname or 'jetmass' in histname:
                        httbar_syst = fttbar[histname][integrate_axes][{'anacat':sum}][::3j]
                        hqcd_syst = fqcd[histname][integrate_axes][{'anacat':sum}][::3j]
                    else:
                        httbar_syst = fttbar[histname][integrate_axes][{'anacat':sum}]
                        hqcd_syst = fqcd[histname][integrate_axes][{'anacat':sum}]
                
                    if 'Up' in syst:
    
                        
                        systvals = np.where(httbar_syst.values()[0] > 0, httbar_syst.values() - nomvals, 0) 
                        systUp2 = systUp2 + systvals * systvals
                    if 'Down' in syst:
                        systvals = np.where(httbar_syst.values()[0] > 0, httbar_syst.values() - nomvals, 0) 
                        systDn2 = systDn2 + systvals * systvals
                
                for syst in fqcd[histname].axes['systematic']:
                    nomvals = h_qcd.values()
                
                    if 'Up' in syst:
                        systvals = np.where(hqcd_syst.values()[0] > 0, hqcd_syst.values() - nomvals, 0) 
                        systUp2 = systUp2 + systvals * systvals
                    if 'Down' in syst:
                        systvals = np.where(hqcd_syst.values()[0] > 0, hqcd_syst.values() - nomvals, 0) 
                        systDn2 = systDn2 + systvals * systvals
            
            
            systUp = np.sqrt(systUp2)
            systDn = np.sqrt(systDn2)

            # stat uncertainties
            # print(intervals.poisson_interval(h_ttbar.values()**2, variances = None, coverage = 0.68))
            # print('nomvals', h_bkg.values())
        
            systUp = systUp + np.sqrt(intervals.poisson_interval(h_ttbar.values(), variances = h_ttbar.variances(), coverage = 0.68)[1]   )
            systUp = systUp + np.sqrt(intervals.poisson_interval(h_qcd.values(), variances = h_ttbar.variances(), coverage = 0.68)[1]     )
            systDn = systDn + np.sqrt(intervals.poisson_interval(h_ttbar.values(), variances = h_qcd.variances(), coverage = 0.68)[0]     )
            systDn = systDn + np.sqrt(intervals.poisson_interval(h_qcd.values(), variances = h_qcd.variances(), coverage = 0.68)[0]       )
                               
            #systUp = systUp + (np.sqrt(h_bkg.values())
            #systDn = systDn + (np.sqrt(h_bkg.values()))
            
            # print(systUp)
            # print(nomvals)
            # print(systDn)

            
            # for i in range(len(nomvals)):
                # print(systDn[i], nomvals[i], systUp[i])
                # print(systDn[i] + np.sqrt(nomvals[i]), nomvals[i], systUp[i] + np.sqrt(nomvals[i]))
                # print()


            
            
            # fig, ax1 = plt.subplots(nrows=2, )


            if histname == 'ht':
                fig, ax1 = plt.subplots(nrows=1)
            else:        
                fig, (ax1, ax2) = plt.subplots(nrows=2, height_ratios=[3, 1], sharex=True)

                
                
            
            text = ''
            if 'cen' in catname:
                text=r'$\bf{central}$'
            elif 'fwd' in catname:
                text=r'$\bf{forward}$'
            
        
        
            
            
            hep.cms.label('Preliminary', data=True, lumi=lumitext, year=yeartext, loc=1, fontsize=20, ax=ax1)
            hep.cms.text(text, loc=2, fontsize=20, ax=ax1)

            if '2016all' in IOV:
                h_bkg = h_qcd*1.5 + h_ttbar
            elif 'all' in IOV:
                normtodata = np.sum(h_data.values())/np.sum(h_bkg.values())
                h_bkg = h_bkg * normtodata

            
            if histname == 'ht':

                
                hep.histplot(h_ttbar, density=False, histtype='fill', color=plotred, label='SM TTbar', ax=ax1)
                hep.histplot(h_sig, density=False, histtype='step', color='black', label='$g_{KK}$ (2 TeV)', ax=ax1)


                ax1.legend(loc=1, fontsize=10)
                ax1.set_yscale('log')
                ax1.set_ylabel('Events/Bin')
    
                ax1.set_ylim(1e-1, 1e4)
                ax1.set_xlim(400, 4400)
                
                savefigname = f'images/{dirstring}/png/kinematics/{IOV}/{histname}_{catname}.png'
                print('saving', savefigname)
                plt.savefig(savefigname)
                plt.savefig(savefigname.replace('png', 'pdf'))




            elif histname == 'jetdy':
            
             # print(np.sum(h_bkg.values())/np.sum(h_sig.values()))

            

                # print (np.abs(h_bkg.axes['jetdy'].edges))
                # print (np.array(h_bkg.values(),dtype='int64'))

                # print( np.repeat(np.abs(h_bkg.axes['jetdy'].centers), np.array(h_bkg.values(),dtype='int64')))

                hbkg_plot = np.repeat(np.abs(h_bkg.axes['jetdy'].centers), np.array(h_bkg.values(),dtype='int64'))


                print(h_bkg.axes['jetdy'].edges[25:])
                print(len(h_bkg.values()[25:]), len(h_bkg.values()[:25]))

                print( np.sum(h_bkg.values()) /  np.sum(h_sig.values()))
                hep.histplot(h_bkg.values()[:25][::-1] + h_bkg.values()[25:], bins=h_bkg.axes['jetdy'].edges[25:], density=False, histtype='fill', color=plotyellow, label='SM QCD', ax=ax1)
                hep.histplot( h_ttbar.values()[:25][::-1] + h_ttbar.values()[25:], bins=h_ttbar.axes['jetdy'].edges[25:], density=False, histtype='fill', color=plotred, label='SM TTbar', ax=ax1)
                hep.histplot( h_data.values()[:25][::-1] + h_data.values()[25:], bins=h_data.axes['jetdy'].edges[25:], density=False, histtype='errorbar', color='black', label='Data', ax=ax1)
                h_sig_plot = h_sig.values()[:25][::-1] + h_sig.values()[25:]
                hep.histplot(11.028627114558727*h_sig_plot, bins=h_sig.axes['jetdy'].edges[25:], density=False, histtype='step', color='black', label='$g_{KK}$ (2 TeV)', ax=ax1)

            
            
                if 'jetmsd' in histname:
                    edges = h_ttbar.axes['jetmass'].edges
                else:
                    edges = h_ttbar.axes[histname.replace('1','')].edges


                height = systUp + systDn
                bottom = h_bkg.values() - systDn
                


             
                
                ax1.bar(x = edges[25:][:-1],
                           height=height[:25][::-1] + height[25:],
                           bottom=bottom[:25][::-1] + bottom[25:],
                           width = np.diff(edges[25:]), align='edge', hatch='///', edgecolor='gray',
                           linewidth=0, facecolor='none', alpha=0.8,
                           zorder=10, label='Stat + Syst')
                
                height_ratio = height/h_bkg.values()
                bottom_ratio = bottom/h_bkg.values()
                ax2.bar(x = edges[25:][:-1],
                           height=0.5*(height_ratio[:25][::-1] + height_ratio[25:]),
                           bottom=0.5*(bottom_ratio[:25][::-1] + bottom_ratio[25:]),
                           width = np.diff(edges[25:]), align='edge',
                           linewidth=0, facecolor='lightgrey', alpha=0.8,
                           zorder=0, label='Stat + Syst')
                ax2.axhline(1, color='black', ls='--')


                ratio_hist = h_data/h_bkg.values()
                hep.histplot(0.5*ratio_hist + 0.5*ratio_hist.values()[::-1], density=False, yerr=ratio_hist.variances(), histtype='errorbar', color='black', ax=ax2)
                
                
                
                
                # print((h_data/h_bkg.values()).values())
                # print(np.average((h_data/h_bkg.values()).values()))
                
                # ax2.bar(x = edges[:-1],
                #            height=1.5,
                #            bottom=0.5,
                #            width = np.diff(edges), align='edge', edgecolor='gray',
                #            linewidth=0, facecolor='gray', alpha=0.8,
                #            zorder=10, label='Syst. Unc.')
                
            
                
                # elif 'jetphi' in histname:
                #     plt.savefig('jetphi_'+IOV+'.pdf')
            
            
                if 'jetmass' in histname:
                    ax1.set_xlim(25,500)
                    ax2.set_xlim(25,500)
                    ax1.set_xlabel('Jet mass [GeV]')
            
                elif 'jetmsd' in histname:
                    ax1.set_xlim(25,500)
                    ax2.set_xlim(25,500)   
            
                elif 'ttbarmass' in histname:
                    ax1.set_xlim(800,6800)
                    ax2.set_xlim(800,6800)
            
            
                
                elif 'jetpt' in histname:
                    ax1.set_xlim(400,2000)
                    ax2.set_xlim(400,2000)
                
                
                elif 'ht' in histname:
                    ax1.set_xlim(500,4500)
                    ax2.set_xlim(500,4500)
                
                elif 'jeteta' in histname:
                    ax1.set_xlim(-2.4,2.4)
                    ax2.set_xlim(-2.4,2.4)
            
                elif 'jetphi' in histname:
                    ax1.set_xlim(-3,3)
                    ax2.set_xlim(-3,3)

                elif 'jetdy' in histname:
                    
                    # ax1.axvline(-1, color='black', ls='--')
                    ax1.axvline(1, color='black', ls='--')
                    ax1.set_xlim(0,3)
                    ax2.set_xlim(0,3)
                    ax2.set_xlabel('|$\Delta y$|')
                
                
                elif 'jety' in histname:
                    ax1.set_xlim(-2.4,2.4)
                    ax2.set_xlim(-2.4,2.4)
                    
                ax1.set_ylim(1e-1, 16000)
                ax2.set_ylim(1-1,1+1)
                
                
                ax1.legend(loc=1, fontsize=10)
                # plt.axvline(-1, color='darkred', ls='--')
                # plt.axvline(1, color='darkred', ls='--')
                # ax1.set_xlim(400,None)
                # ax1.set_yscale('log')
                ax1.set_xlabel('')
                
                ax1.set_ylabel('Events/Bin')
                ax2.set_ylabel('Data/MC')
    
    
                if histname == 'jetmass':
                    ax2.set_xlabel('Jet mass [GeV]')
                
                
                savefigname = f'images/{dirstring}/png/kinematics/{IOV}/{histname}_{catname}.png'
                print('saving', savefigname)
                plt.savefig(savefigname)
                plt.savefig(savefigname.replace('png', 'pdf'))

            
            else:


                hep.histplot(h_bkg, density=False, histtype='fill', color=plotyellow, label='SM QCD', ax=ax1)
                hep.histplot(h_ttbar, density=False, histtype='fill', color=plotred, label='SM TTbar', ax=ax1)
                hep.histplot(h_data, density=False, histtype='errorbar', color='black', label='Data', ax=ax1)
                hep.histplot(h_sig, density=False, histtype='step', color='black', label='$g_{KK}$ (2 TeV)', ax=ax1)

                
            
                if 'jetmsd' in histname:
                    edges = h_ttbar.axes['jetmass'].edges
                else:
                    edges = h_ttbar.axes[histname.replace('1','')].edges


                height = systUp + systDn
                bottom = h_bkg.values() - systDn
                


             
                
                ax1.bar(x = edges[:-1],
                           height=height,
                           bottom=bottom,
                           width = np.diff(edges), align='edge', hatch='///', edgecolor='gray',
                           linewidth=0, facecolor='none', alpha=0.8,
                           zorder=10, label='Stat + Syst')
                
                height_ratio = height/h_bkg.values()
                bottom_ratio = bottom/h_bkg.values()
                ax2.bar(x = edges[:-1],
                           height=height_ratio,
                           bottom=bottom_ratio,
                           width = np.diff(edges), align='edge',
                           linewidth=0, facecolor='lightgrey', alpha=0.8,
                           zorder=0, label='Stat + Syst')
                ax2.axhline(1, color='black', ls='--')


                ratio_hist = h_data/h_bkg.values()
                hep.histplot(ratio_hist, density=False, histtype='errorbar', color='black', ax=ax2)
                
                
                
                
                # print((h_data/h_bkg.values()).values())
                # print(np.average((h_data/h_bkg.values()).values()))
                
                # ax2.bar(x = edges[:-1],
                #            height=1.5,
                #            bottom=0.5,
                #            width = np.diff(edges), align='edge', edgecolor='gray',
                #            linewidth=0, facecolor='gray', alpha=0.8,
                #            zorder=10, label='Syst. Unc.')
                
            
                
                # elif 'jetphi' in histname:
                #     plt.savefig('jetphi_'+IOV+'.pdf')
            
            
                if 'jetmass' in histname:
                    ax1.set_xlim(25,500)
                    ax2.set_xlim(25,500)
                    ax1.set_xlabel('Jet mass [GeV]')
            
                elif 'jetmsd' in histname:
                    ax1.set_xlim(25,500)
                    ax2.set_xlim(25,500)   
            
                elif 'ttbarmass' in histname:
                    ax1.set_xlim(800,6800)
                    ax2.set_xlim(800,6800)
            
            
                
                elif 'jetpt' in histname:
                    ax1.set_xlim(400,2000)
                    ax2.set_xlim(400,2000)
                
                
                elif 'ht' in histname:
                    ax1.set_xlim(500,4500)
                    ax2.set_xlim(500,4500)
                
                elif 'jeteta' in histname:
                    ax1.set_xlim(-2.4,2.4)
                    ax2.set_xlim(-2.4,2.4)
            
                elif 'jetphi' in histname:
                    ax1.set_xlim(-3,3)
                    ax2.set_xlim(-3,3)

                elif 'jetdy' in histname:
                    
                    # ax1.axvline(-1, color='black', ls='--')
                    ax1.axvline(1, color='black', ls='--')
                    ax1.set_xlim(0,3)
                    ax2.set_xlim(0,3)
                    ax2.set_xlabel('|$\Delta y$|')
                
                
                elif 'jety' in histname:
                    ax1.set_xlim(-2.4,2.4)
                    ax2.set_xlim(-2.4,2.4)
                    
                
                
                
                if 'jetmsd' in histname:
                    ax2.set_xlabel('$m_{t}$ [GeV]')
                    ax1.set_ylim(1e-1, 0.3*np.max(h_data.values()))


                else:
                    ax1.set_yscale('log')
                    ax1.set_ylim(1e-1, 1e7)
                    
                
                ax2.set_ylim(1-1,1+1)
                
                
                ax1.legend(loc=1, fontsize=10)
                ax1.set_xlabel('')
                
                ax1.set_ylabel('Events/Bin')
                ax2.set_ylabel('Data/MC')
    
    
                if histname == 'jetmass':
                    ax2.set_xlabel('Jet mass [GeV]')
                
                
                savefigname = f'images/{dirstring}/png/kinematics/{IOV}/{histname}_{catname}.png'
                print('saving', savefigname)
                plt.savefig(savefigname)
                plt.savefig(savefigname.replace('png', 'pdf'))



plotKinematics(IOV, dataset='TTbar', dirstring=savedirstring)


# plotSystematics(IOV, dataset='RSGluon5000', dirstring=savedirstring)
# plotSystematics(IOV, dataset='RSGluon2000', dirstring=savedirstring)
# plotSystematics(IOV, dataset='TTbar', dirstring=savedirstring)



# plotSystematics(IOV, dataset='TTbar', dirstring=savedirstring)



   
