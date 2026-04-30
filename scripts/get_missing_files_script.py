# run missing files

import os, sys

directory = '../outputs/an_v6/'

print('checking missing files from directory', directory, '\n')



if (len(sys.argv) > 1) and (sys.argv[1] in ['2016', '2016APV', '2017', '2018']):
    
    year = sys.argv[1]

else:

    year = '2016'


files = []
for file in os.listdir(directory):
    if 'coffea' in file:
        files.append(file)


years = ['2016APV', '2016', '2017', '2018']
jetht = {
    '2016APV': [
        'JetHT_2016APVB',
        'JetHT_2016APVC',
        'JetHT_2016APVD',
        'JetHT_2016APVE',
        'JetHT_2016APVF',
    ],
    '2016': [
        'JetHT_2016F',
        'JetHT_2016G',
        'JetHT_2016H',
    ],
    '2017': [
        'JetHT_2017B',
        'JetHT_2017C',
        'JetHT_2017D',
        'JetHT_2017E',
        'JetHT_2017F',
    ],
    '2018': [
        'JetHT_2018A',
        'JetHT_2018B',
        'JetHT_2018C',
        'JetHT_2018D',
    ]
}

datasets = ['JetHT', 'QCD', 'TTbar', 'RSGluon', 'ZPrimeDM', 'ZPrime10', 'ZPrime30', 'ZPrime1']


commands = []
missing = True
for dataset in datasets:
    
    if 'JetHT' in dataset:
        command = 'python ttbaranalysis.py --dask --iov {0} -d JetHT'.format(year)
        for era in jetht[year]:
            missing = True
            fname = era + '.coffea'
            for fsaved in files:
                if fsaved == fname:
                    missing = False
            if missing:
                command += ' --era {0}'.format(era[-1:]) 
        if '--era' in command:
            commands.append(command)
    
    elif 'RSGluon' in dataset:
        command = 'python ttbaranalysis.py --dask -n --iov {0} -d RSGluon'.format(year)
        masses = [10,15,20,25,30,35,40,45,50,55,60]
        
        for mass in masses:
            missing = True

            fname = dataset + str(int(mass*100)) + '_' + year + '.coffea'
            for fsaved in files:
                if fsaved == fname:
                    missing = False
            if missing:
                command += ' -m {0}'.format(int(mass*100)) 
        if '-m' in command:
            commands.append(command)
            
    elif 'ZPrimeDM' == dataset:
        command = 'python ttbaranalysis.py --dask -n --iov {0} -d ZPrimeDM'.format(year)
        masses = [10,15,20,25,30,35,40,45,50]
        
        for mass in masses:
            missing = True
            fname = dataset[:-2] + str(int(mass*100)) + '_DM_' + year + '.coffea'
            for fsaved in files:
                if fsaved == fname:
                    missing = False
            if missing:
                command += ' -m {0}'.format(int(mass*100)) 
        if '-m' in command:
            commands.append(command)
            
    elif 'ZPrime10' == dataset:
        command = 'python ttbaranalysis.py --dask -n --iov {0} -d ZPrime10'.format(year)
        masses = [10,12,14,16,18,20,25,30,35,40,45,50,60,70]
        
        for mass in masses:
            missing = True
            fname = dataset[:-2] + str(int(mass*100)) + '_10_' + year + '.coffea'
            for fsaved in files:
                if fsaved == fname:
                    missing = False
            if missing:
                command += ' -m {0}'.format(int(mass*100)) 
        if '-m' in command:
            commands.append(command)
            
    elif 'ZPrime30' == dataset:
        command = 'python ttbaranalysis.py --dask -n --iov {0} -d ZPrime30'.format(year)
        masses = [10,12,14,16,18,20,25,30,35,40,45,50,60,70]
        
        for mass in masses:
            missing = True
            fname = dataset[:-2] + str(int(mass*100)) + '_30_' + year + '.coffea'
            for fsaved in files:
                if fsaved == fname:
                    missing = False
            if missing:
                command += ' -m {0}'.format(int(mass*100)) 
        if '-m' in command:
            commands.append(command)
            
    elif 'ZPrime1' == dataset:
        command = 'python ttbaranalysis.py --dask -n --iov {0} -d ZPrime1'.format(year)
        masses = [10,12,14,16,18,20,25,30,35,40,45]
        
        for mass in masses:
            missing = True
            fname = dataset[:-1] + str(int(mass*100)) + '_1_' + year + '.coffea'
            for fsaved in files:
                if fsaved == fname:
                    missing = False
            if missing:
                command += ' -m {0}'.format(int(mass*100)) 
        if '-m' in command:
            commands.append(command)
    
    elif 'TTbar' in dataset:
        command = 'python ttbaranalysis.py --dask --iov {0} -d TTbar'.format(year)
        for pt in ['700to1000', '1000toInf']:
            missing = True
            fname = dataset + '_' + year + '_' + pt + '.coffea'
            for fsaved in files:
                if fsaved == fname:
                    missing = False
            if missing:
                command += ' --pt {0}'.format(pt) 
        if '--pt' in command:
            commands.append(command)
            
            
        
    else:
        
        command = 'python ttbaranalysis.py --dask --iov {0} -d {1}'.format(year, dataset)
        
        missing = True
        fname = dataset + '_' + year + '.coffea'
        
        for fsaved in files:
            if fsaved == fname:
                missing = False
        if missing:
            commands.append(command)
                        
                    
if len(commands) < 1:
    print('No files missing!')



for command in commands:
    print(command)
    print()
