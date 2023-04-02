import pandas as pd
import numpy as np
import glob
from unidecode import unidecode
import requests, contextlib, re, os
import geobr
import geopandas as gpd
from matplotlib import pyplot as plt
from matplotlib import colors
import esda
from esda.moran import Moran_Local
from shapely.geometry import Point, Polygon
import libpysal as lps
import geopandas as gpd
from splot.esda import moran_scatterplot, plot_moran, lisa_cluster

# link='https://www.ibge.gov.br/estatisticas/economicas/agricultura-e-pecuaria/9117-producao-agricola-municipal-culturas-temporarias-e-permanentes.html?=&t=resultados'
#
# text_soup = soup(requests.get(link).text, "lxml-xml")
# for link in text_soup.find_all('a'):
#     print(link.get('href'))
# 'Cebola',
crops = ['Soja (em grão)', 'Milho (em grão)', 'Trigo (em grão)', 'Arroz (em casca)']
# crops = [nan, 'Laranja', 'Café (em grão) Canephora', 'Aveia (em grão)',
#          'Algodão arbóreo (em ca...', 'Cacau (em amêndoa)', 'Malva (fibra)',
#          'Café (em grão) Arábica', 'Mandioca', 'Pêssego', 'Melancia',
#          'Café (em grão) Total', 'Castanha de caju', 'Caju', 'Marmelo',
#          'Noz (fruto seco)', 'Maçã', 'Amendoim (em casca)',
#          'Fumo (em folha)', 'Caqui', 'Sisal ou agave (fibra)', 'Mamão',
#          'Cebola', 'Abacate', 'Triticale (em grão)', 'Urucum (semente)',
#          'Alfafa fenada', 'Manga', 'Girassol (em grão)', 'Milho (em grão)',
#          'Batata-inglesa', 'Alho', 'Sorgo (em grão)', 'Tomate', 'Limão',
#          'Palmito', 'Linho (semente)', 'Borracha (látex líquido)',
#          'Tungue (fruto seco)', 'Erva-mate (folha verde)', 'Arroz (em casca)',
#          'Cana para forragem', 'Algodão herbáceo (em c...', 'Ervilha (em grão)',
#          'Dendê (cacho de coco)', 'Figo', 'Melão', 'Total', 'Tangerina',
#          'Maracujá', 'Cevada (em grão)', 'Soja (em grão)', 'Goiaba',
#          'Centeio (em grão)', 'Pimenta-do-reino', 'Uva',
#          'Borracha (látex coagul...', 'Azeitona', 'Cana-de-açúcar',
#          'Juta (fibra)', 'Coco-da-baía*', 'Trigo (em grão)', 'Feijão (em grão)',
#          'Pera', 'Mamona (baga)', 'Guaraná (semente)', 'Abacaxi*', 'Batata-doce',
#          'Banana (cacho)', 'Açaí', 'Chá-da-índia (folha ve...', 'Fava (em grão)',
#          'Rami (fibra)']
read_csv = True
read_shape = False
directory = '/home/eduardo/Documents/Eduardo/MBA_DS/base_dados'
p = 0.05
weights = 'Queen'  # Distance


def define_nivel(string):
    # counter
    k = 0
    nivel = 0
    # loop for search each index
    if isinstance(string, str):
        for i in string:
            # Verifica se espaco em branco
            if i == " ":
                k += 1
            # Pare quando encontrar o primeiro caractere
            else:
                break
        if k == 0:
            # Unidade Federal
            nivel = 1
        elif k == 2:
            # Mesoregiao
            nivel = 2
        elif k == 4:
            # Microregiao
            nivel = 3
        elif k == 6:
            # Municipio
            nivel = 4
        else:
            None
    else:
        nivel = 0
    return nivel


def define_subgrupos(n):
    nivel_dict = {1: 'UF', 2: 'Mesoregiao', 3: 'Microregiao', 4: 'Municipio'}
    regioes = {i: sheet['Local'][i] for i, nivel in enumerate(sheet['Nivel']) if nivel == n}
    indices_regioes = list(regioes.keys())
    sheet[nivel_dict[n]] = np.nan
    for i in range(len(sheet['Local'])):
        if i >= indices_regioes[0]:
            idx_regiao = [r for r in indices_regioes if r <= i][-1]
            regiao = regioes[idx_regiao]
        else:
            regiao = np.nan
        sheet.loc[i, nivel_dict[n]] = regiao


# Dados de produtividade por estado e cultura


def lisa_cluster_manual(gdf, Moran_local, p, figsize, fig_path,
                        cultura, weights, HH_only=False
                        ):
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    sig = 1 * (Moran_local.p_sim < p)
    hotspot = 1 * (sig * Moran_local.q == 1)
    coldspot = 3 * (sig * Moran_local.q == 3)
    doughnut = 2 * (sig * Moran_local.q == 2)
    diamond = 4 * (sig * Moran_local.q == 4)
    spots = hotspot + coldspot + doughnut + diamond
    spot_labels = ['0 ns', '1 HH', '2 LH', '3 LL', '4 HL']
    labels = [spot_labels[i] for i in spots]
    if HH_only:
        hmap = colors.ListedColormap(['lightgrey', 'red', 'lightgrey', 'lightgrey', 'lightgrey', ])
    else:
        hmap = colors.ListedColormap(['lightgrey', 'red', 'lightblue', 'blue', 'pink'])

    ax = gdf.assign(cl=labels).plot(column='cl', categorical=True, \
                                    k=2, cmap=hmap, linewidth=0.1, ax=ax, \
                                    edgecolor='white', legend=True)
    ax.set_axis_off()
    # ax.legend(fontsize=24)
    plt.title('{} \n {} \n {}'.format(crop, col, 'Mapa de Hot e Cold spots'), fontsize=36)

    f.tight_layout()

    # Display the figure
    plt.savefig('{}/{}_{}_LISA_clusters.png'.format(fig_path, cultura, weights), bbox_inches='tight')
    plt.close()


if not read_csv:
    crops_df = pd.DataFrame()
    # for file in directory:
    for file in glob.glob(directory + '/*.xlsx'):
        print(file)
        # Cabecalho comeca na quinta linha
        excel_sheets = pd.read_excel(file, sheet_name=None, header=4)

        all_sheets = []
        full_table = []
        n_colunas = 6
        for name, sheet in excel_sheets.items():
            if len(sheet.columns) == n_colunas:
                cultura = name.split(';')[-1].lstrip()
                print(cultura)
                sheet['Cultura'] = cultura
                # Renomeando primeira coluna de nan para Local
                sheet.rename(columns={sheet.columns[0]: "Local"}, inplace=True)
                sheet['Nivel'] = list(map(define_nivel, sheet['Local']))
                sheet['Local'] = [local.lstrip() for local in sheet['Local']]

                meso = np.nan
                micro = np.nan
                niveis_hierarquicos = ['UF', 'Mesoregiao', 'Microregiao', 'Municipio']
                for i in range(len(niveis_hierarquicos)):
                    define_subgrupos(n=i + 1)

                all_sheets.append(sheet)

        full_table = pd.concat(all_sheets)
        full_table.reset_index(inplace=True, drop=True)

        full_table.columns = [unidecode(col) for col in full_table.columns]
        crops_df = pd.concat([crops_df, full_table])

        crops_df.to_csv(directory + '/crops.csv', index=True)
else:
    crops_df = pd.read_csv(directory + '/crops.csv')

estados = ['Acre', 'Alagoas', 'Amapá', 'Amazonas', 'Bahia', 'Ceará',
           'Distrito Federal', 'Espírito Santo', 'Goiás', 'Maranhão',
           'Mato Grosso', 'Mato Grosso do Sul', 'Minas Gerais', 'Pará', 'Paraíba',
           'Paraná', 'Pernambuco', 'Piauí', 'Rio de Janeiro',
           'Rio Grande do Norte', 'Rio Grande do Sul', 'Rondônia', 'Roraima',
           'Santa Catarina', 'São Paulo', 'Sergipe', 'Tocantins'
           ]
siglas = ['AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA',
          'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN',
          'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO'
          ]
estados_dict = dict(zip(estados, siglas))

# Filtrando base de dados para remover entradas nao relacionadas
crops_df = crops_df.loc[crops_df['UF'].isin(estados)]
crops_df = crops_df.replace(['-', '..', '...'], np.nan)
# Adicionando siglas de estados
crops_df['UF_sigla'] = [estados_dict[v] for v in crops_df['UF']]

columns_to_convert = [
    'Area plantada (Hectares)',
    'Area colhida (Hectares)',
    'Quantidade produzida (Toneladas)',
    'Rendimento medio da producao (Quilogramas por Hectare)',
    'Valor da producao (Mil Reais)'
]

columns_dict = {column: column[:3]+ '_' + str(k)
                for k, column in enumerate(crops_df.columns)
                }
crops_df.columns = [value for value in columns_dict.values()]

columns_to_convert_idx = [columns_dict[col] for col in columns_to_convert]

for col in columns_to_convert_idx:
    crops_df[col] = crops_df[col].astype(float)
shape_path = directory + '/shapes'

if not read_shape:
    geodata = geobr.read_municipality(code_muni="all", year=2019)
    geodata.to_file(directory+"/geodata.shp")
# shape_filename = shape_path + '/crops.shp'
if not crops:
    crops = set(crops_df[columns_dict['Cultura']])
n = 1
for crop in crops:
    print('Preparing: {} \n '
          'crop {} of {}'.format(crop, n, len(crops)))
    n += 1
    shape_path = directory + '/shapes/' + crop
    shape_filename = shape_path + '/' + crop + '.shp'
    if read_shape:
        culturas_df = gpd.read_file(shape_filename)
    else:
        culturas_df = crops_df.loc[
            (crops_df[columns_dict['Cultura']] == crop) &
            (crops_df[columns_dict['Nivel']] == 4)
            ]
        # culturas_df = culturas_df.loc[culturas_df['Municipio'].notnull()]
        # geodata.to_csv(directory+'/geodata.csv')
        # Padronizando regras de nomenclatura entre os dois datasets
        geodata.loc[:, ('name_muni')] = \
            geodata.loc[:,('name_muni')].str.lower().apply(
                unidecode).str.replace('-', ' ')

        culturas_df.loc[:, (columns_dict['Municipio'])] = \
            culturas_df.loc[:, (columns_dict['Municipio'])].str.lower().apply(
                unidecode).str.replace('-', ' ')

        culturas_df = pd.merge(
            geodata, culturas_df,
            right_on=[columns_dict['UF_sigla'], columns_dict['Municipio']],
            left_on=['abbrev_state', 'name_muni'],
            how='outer'
        )

        culturas_df = gpd.GeoDataFrame(culturas_df)

        if not os.path.exists(shape_path):
            os.makedirs(shape_path)
        culturas_df.to_file(shape_filename)

    # Coluna definida para analise espacial
    col = columns_dict['Rendimento medio da producao (Quilogramas por Hectare)']

    column_path = shape_path + "/" + str(col)
    if not os.path.exists(column_path):
        os.makedirs(column_path)
    missing_kwds = dict(color='grey', label='No Data')
    ax = culturas_df.plot(column=col, legend=True,
                          scheme="quantiles",
                          figsize=(15, 15),
                          legend_kwds=dict(loc='center left'),
                          missing_kwds=missing_kwds
                          )
    ax.set_axis_off()
    # ax.legend(fontsize=24)
    plt.title('{} \n {}'.format(crop, col), fontsize=36)
    plt.savefig(column_path + '/' + crop + '_' + weights + '.png', dpi=200)

    if culturas_df[col].count() > 5:

        # Filtra dados ao nivel de municipio
        cultura_geodf = culturas_df.loc[
            (culturas_df[columns_dict['Nivel']] == 4),
            ['geometry', 'Municipio', col]
        ]
        # cultura_geodf = culturas_df.loc[
        # (culturas_df['Nivel']==4), ['geometry', 'Municipio', col]
        # ].fillna(0)
        cultura_geodf = cultura_geodf[['geometry', col]]
        print('Geodataframe pronto')
        if weights == "Queen":
            w = lps.weights.Queen.from_dataframe(cultura_geodf)
            w.transform = 'r'
            print('Pesos baseados em  movimento Queen preparados')
        elif weights == 'Distance':
            w = lps.weights.DistanceBand.from_dataframe(cultura_geodf, threshold=30)
            print("Pesos baseados em Distancia preparados")

        cultura_geodf[col].fillna(0, inplace=True)
        y = cultura_geodf[col]

        ylag = lps.weights.lag_spatial(w, y)
        mi = esda.moran.Moran(y, w)

        print(crop, '\n',
              col, '\n',
              "Moran's I:", mi.I, '\n',
              "Moran's p-value:", mi.p_norm, '\n',
              "Moran's z-score:", mi.z_norm
              )

        try:
            moran_loc = Moran_Local(y, w)

            # # Generates Moran's Global Scatterplot with data histogram
            # plot_moran(mi, zstandard=True, figsize=(10, 4))
            # plt.savefig(folder_plots + side + '_' + label + '_' + variable + 'reference_distribution_moran_scatterplot_global_AC' + '.png')
            # plt.close()

            # Generates Moran's Global Scatterplot
            fig, ax = plt.subplots(figsize=(21, 10))
            moran_scatterplot(mi, aspect_equal=True, zstandard=True, ax=ax)
            plt.suptitle(crop + ', ' + col + '\n' + "Moran's p-value:" + str(mi.p_norm) + '\n' + "Moran's z-score:" + str(mi.z_norm))
            plt.savefig(column_path + '/' + crop + '_' + weights + '_' + 'moran_scatterplot_global_AC' + '.png', bbox_inches='tight')
            plt.close()

            # Generates Moran Local Scatterplot
            fig, ax = moran_scatterplot(moran_loc, p=p)
            ax.set_xlabel(col)
            ax.set_ylabel('Spatial Lag of {}'.format(col))
            ax.text(1.95, 0.5, "HH", fontsize=25)
            ax.text(1.95, -1, "HL", fontsize=25)
            ax.text(-1, 0.5, "LH", fontsize=25)
            ax.text(-1, -1, "LL", fontsize=25)
            plt.savefig(column_path + '/' + crop + '_' + weights + '_' + 'moran_scatterplot_local_AC' + '.png')
            plt.close()

            lisa_cluster_manual(gdf=cultura_geodf, Moran_local=moran_loc, p=p,
                                figsize=(15, 15), fig_path=column_path,
                                weights=weights, cultura=crop
                                )
        except Exception as e:
            print(e)
            pass
