import os
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from statistical_evolution import slices2evol, evolrhis


rcParams['lines.linewidth'] = 0.5
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Verdana'

data_type = 'meteorologicos' # chuva, meteorologicos, vazao, qualidade
t = [0, 0] #backwards slice of 10 years [monthly scale, annual scale], [0, 0] for complete series
start_slice = [60, 5]
years = 10
masterdict = {}
folder = '%s_teste'%data_type

if t[0] == 0:
    writer1 = pd.ExcelWriter('RHIS_series_completas_%s_MENSAL.xlsx'%data_type)
    writer2 = pd.ExcelWriter('RHIS_series_completas_%s_ANUAL.xlsx'%data_type)
if t[0] != 0:
    writer1 = pd.ExcelWriter('RHIS_last%syears_%s_MENSAL.xlsx'%(years, data_type))
    writer2 = pd.ExcelWriter('RHIS_last%syears_%s_ANUAL.xlsx'%(years, data_type))

path = "C:\\Users\\Dilson\\Documents\\Python Scripts\\Projeto_Paranapanema\\%s"%folder


for file in os.listdir(path):

    if file.endswith('.txt'):
        if data_type == 'chuva':
            cod = file[:-4].split('_')[-1]

            try:
                chuva_info = pd.read_csv('chuva_DAEE_Tabela_Estacoes.txt', sep="\t", index_col=1)
                lat = chuva_info.loc[cod, 'Latitude']
                lon = chuva_info.loc[cod, 'Longitude']
            except KeyError:

                try:
                    chuva_info = pd.read_csv('chuva_Estacoes_Juntadas.txt', sep="\t", index_col=0)
                    lat = chuva_info.loc[cod, 'lat']
                    lon = chuva_info.loc[cod, 'lon']
                except KeyError:

                    try:
                        chuva_info = pd.read_csv('chuva_hidroweb_Tabela_Estacoes.txt', sep="\t", index_col=0)
                        lat = chuva_info.loc[cod, 'Latitude']
                        lon = chuva_info.loc[cod, 'Longitude']
                    except KeyError:
                        lat = np.nan
                        lon = np.nan

        if data_type == 'meteorologicos':
            cod = file[:-4].split('_')[2]

            try:
                meteo_info = pd.read_csv('tabela INMET.csv', sep="\t", index_col=0)
                lat = meteo_info.loc[cod, 'LATITUDE']
                lon = meteo_info.loc[cod, 'LONGITUDE']
            except KeyError:
                lat = np.nan
                lon = np.nan

        if data_type == 'chuva':
            units = ['mm/mês', 'mm/ano']
            dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
            df = pd.read_csv('%s\\%s'%(path, file), sep="\t", parse_dates=True, date_parser=dateparse, header=[0], index_col=0)
            
            monthly_0 = df.resample('M').sum()
            monthly = pd.DataFrame(monthly_0['Chuva'])
            annual_0 = df.resample('Y').sum()
            annual = pd.DataFrame(annual_0['Chuva'])
            colors = ['0.3',]
            trendcolor = ['k',]
            cyclecolor = ['k',]
            figure = (10, 5)
            sharey = False

        if data_type == 'meteorologicos':
            units = {'T_Ar': '°C', 'Umidade': '%', 'Pressao': 'hPa', 'Nebulosidade': '%', 'Radiacao': 'W/m²'}
            colors = ['r', 'b', 'c', 'm', 'g']
            trendcolor = colors
            cyclecolor = colors
            figure = (10, 8)
            sharey = 'row'
            dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
            df = pd.read_csv('%s\\%s'%(path, file), sep="\t", parse_dates=True, date_parser=dateparse, header=[0], index_col=0)
            monthly = df.resample('M').mean()
            annual = df.resample('Y').mean()
        
        ts = [monthly, annual]
        titles = ['MENSAL', 'ANUAL']
         
        file_dict = {'MENSAL':{}, 'ANUAL':{}}
 
        if len(annual) >= 5:
            
            fig, axes = plt.subplots(len(monthly.columns) + 1, 2, sharey=sharey, sharex='col', figsize=figure)

            for i in range(len(ts)):
                names_var = []
                rhis = []

                for j in range(len(monthly.columns)):
                    slices = slices2evol(ts[i][monthly.columns[j]][::-1], start_slice[i])
                    rand = np.array(evolrhis(slices, 'randomness'))
                    homo = np.array(evolrhis(slices, 'homogeneity'))
                    ind = np.array(evolrhis(slices, 'independence'))
                    stat = np.array(evolrhis(slices, 'stationarity'))
                
                    dict2 = {monthly.columns[j]: ts[i][monthly.columns[j]]}
                    df2 = pd.DataFrame(dict2)
                    
                    tail = np.ones(start_slice[i] - 1)*np.nan
                    rand_inv = rand[::-1]
                    homo_inv = homo[::-1]
                    ind_inv = ind[::-1]
                    stat_inv = stat[::-1]
                    df2['R'] = np.concatenate((rand_inv, tail))
                    df2['H'] = np.concatenate((homo_inv, tail))
                    df2['I'] = np.concatenate((ind_inv, tail))
                    df2['S'] = np.concatenate((stat_inv, tail))
                    df2['trend'] = df2['H'] + df2['S']
                    df2['cycle'] = df2['R'] + df2['I']
                    
                    file_dict[titles[i]][monthly.columns[j]] = [df2['R'][t[i]], 
                                   df2['H'][t[i]], df2['I'][t[i]], df2['S'][t[i]],
                                   df2['trend'][t[i]], df2['cycle'][t[i]],
                                   '%s-%s'%(df2.index[t[i]].year, df2.index[-1].year)]
                   
                    
                    title = "\n".join(textwrap.wrap('%s_%s'%(file[:-4], titles[i]), 43))
                    axes[j, i].plot(ts[i][monthly.columns[j]], color=colors[j], linewidth=1.0)
                    axes[j, i].grid(True, linewidth=0.5, linestyle='--', axis='both')
                    axes[0, i].set_title(title, fontsize=10)
                    
                    axes[len(monthly.columns), i].plot(df2['trend'], color=trendcolor[j], linewidth=1.0, label='trend')
                    axes[len(monthly.columns), i].plot(df2['cycle'], color=cyclecolor[j], linestyle='--', linewidth=1.0, label='cycle')

                    if data_type == 'chuva':
                        axes[len(monthly.columns), i].fill_between(df2.index, 0.0, 0.1, color='r', alpha=0.5)
                        axes[len(monthly.columns), i].fill_between(df2.axes[0], 0.1, 1.05, color='y', alpha=0.5)
                        axes[len(monthly.columns), i].fill_between(df2.axes[0], 1.05, 2.0, color='g', alpha=0.5)

                    if data_type == 'meteorologicos':
                        axes[len(monthly.columns), i].fill_between(df2.index, 0.0, 0.1, color='0.6', alpha=0.2)
                        axes[len(monthly.columns), i].fill_between(df2.axes[0], 0.1, 1.05, color='0.8', alpha=0.2)
                        axes[len(monthly.columns), i].fill_between(df2.axes[0], 1.05, 2.0, color='0.9', alpha=0.2)
    
                    axes[len(monthly.columns), i].set_xlabel('ano')
                    axes[len(monthly.columns), i].set_xlim(df2.index[0], df2.index[-1])
                    axes[len(monthly.columns), i].set_ylim(0, 2)

                    try:
                        axes[j, i].set_ylim(0, df2[monthly.columns[j]].max() + 0.1*df2[monthly.columns[j]].max())
                    except ValueError:
                        pass
                    axes[len(monthly.columns), 0].legend(loc=2, framealpha=0.5, ncol=len(monthly.columns), fontsize=8)

                    if data_type == 'chuva':
                        axes[j, i].set_ylabel(units[i], fontsize=10)

                    if data_type == 'meteorologicos':
                        axes[j, i].set_ylabel('%s %s'%(monthly.columns[j], units[monthly.columns[j]]), fontsize=10)
                
                
                file_dict[titles[i]]['Latitude'] = lat
                file_dict[titles[i]]['Longitude'] = lon
            masterdict[file[:-4]] = file_dict

            plt.tight_layout()
            plt.savefig('RHIS_%s.png'%file[:-4], dpi=300)
            plt.show()

writers = [writer1, writer2]

stations = list(masterdict.keys())

for i in range(len(stations)):
    scales = list(masterdict[stations[i]].keys())

    for j in range(len(scales)):
        varlist = list(masterdict[stations[i]][scales[j]].keys())[:-2]
        
        for var in varlist:
            frame = dict(estacao=[], R=[], H=[], I=[], S=[],
                      tendencia=[], sazonalidade=[], periodo=[], latitude=[], longitude=[])
            keys = list(frame.keys())[1:-2]
            frame['estacao'].append(stations[i])
            frame['latitude'].append(masterdict[stations[i]][scales[j]]['Latitude'])
            frame['longitude'].append(masterdict[stations[i]][scales[j]]['Longitude'])
            for k in range(len(keys)):
                frame[keys[k]].append(masterdict[stations[i]][scales[j]][var][k])   
            df3 = pd.DataFrame(frame)
            
            if i == 0:
                df3.to_excel(writers[j], sheet_name=var, header=True)
            if i != 0 and var in writers[j].sheets:
                df3.to_excel(writers[j], sheet_name=var, header=False, startrow= i + 1)
            if i != 0 and var not in writers[j].sheets:
                df3.to_excel(writers[j], sheet_name=var, header=True, startrow= 0)
            
writer1.save()
writer2.save()         
            