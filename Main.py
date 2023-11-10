import pandas as pd
import numpy as np
import glob
from unidecode import unidecode
import requests, contextlib, re, os
import geobr
import geopandas as gpd
from matplotlib import pyplot as plt
from matplotlib import colors, ticker
from matplotlib.colors import ListedColormap
import esda
from esda.moran import Moran_Local
from shapely.geometry import Point, Polygon
import libpysal as lps
import geopandas as gpd
from splot.esda import moran_scatterplot
import seaborn as sns
from crops import crops
# import locale
# locale.setlocale(locale.LC_ALL, 'de_DE')
# link='https://www.ibge.gov.br/estatisticas/economicas/agricultura-e-pecuaria/9117-producao-agricola-municipal-culturas-temporarias-e-permanentes.html?=&t=resultados'


# Flags
read_csv = True
read_shape = False
prepare_summary = False

# Folders Locations
root_directory = '/home/eduardo/Documents/Eduardo/MBA_DS'
# climate_directory = f'{root_directory}clima'
# directory = f'{root_directory}base_dados'
# soil_directory = f'{root_directory}solos'


# Parameters and variables
var_nome_original = 'Produtividade (kg/ha)'
p = 0.05
weights = 'Queen'  # 'Distance'

analysis_params = {
    'diretorio':root_directory,
    'variavel':var_nome_original,
    'p-value':p,
    'weight':weights
}

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


def define_subgrupos(n, sheet):
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


# Data preparation
def prepare_dict_estados_regioes():
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
    regioes = ['N', 'NE', 'N', 'N', 'NE', 'NE', 'CO', 'SE', 'CO', 'N',
               'CO', 'CO', 'SE', 'N', 'NE', 'S', 'NE', 'NE', 'SE', 'NE',
               'S', 'N', 'N', 'S', 'SE', 'NE', 'CO'
               ]

    estados_dict = dict(zip(estados, siglas))
    regioes_dict = dict(zip(estados, regioes))

    return estados_dict, regioes_dict


def prepare_crops_dataframe():
    global analysis_params
    directory = f'{analysis_params["diretorio"]}/base_dados'
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
                    define_subgrupos(n=i + 1, sheet=sheet)

                all_sheets.append(sheet)

        full_table = pd.concat(all_sheets)
        full_table.reset_index(inplace=True, drop=True)

        full_table.columns = [unidecode(col) for col in full_table.columns]
        crops_df = pd.concat([crops_df, full_table])

        crops_df.to_csv(directory + '/crops.csv', index=True)

        return crops_df


def format_crops_dataframe(crops_df):
    # Filtrando base de dados para remover entradas nao relacionadas
    estados_dict, regioes_dict = prepare_dict_estados_regioes()
    estados = list(estados_dict.keys())
    crops_df = crops_df.loc[crops_df['UF'].isin(estados)]
    crops_df = crops_df.replace(['-', '..', '...'], np.nan)
    # Adicionando siglas de estados
    crops_df['UF_sigla'] = [estados_dict[v] for v in crops_df['UF']]
    crops_df['Regiao'] = [regioes_dict[v] for v in crops_df['UF']]
    columns_to_rename_dict = {
        'Area plantada (Hectares)': 'Area plantada (ha)',
        'Area colhida (Hectares)': 'Area colhida (ha)',
        'Quantidade produzida (Toneladas)': 'Producao (ton)',
        'Rendimento medio da producao (Quilogramas por Hectare)': 'Produtividade (kg/ha)',
        'Valor da producao (Mil Reais)': 'Receita (Mil Reais)'
    }

    crops_df.rename(columns=columns_to_rename_dict, inplace=True)

    columns_to_convert = [val for val in columns_to_rename_dict.values()]

    for col in columns_to_convert:
        crops_df[col] = crops_df[col].astype(float)

    return crops_df


def prepare_climate_data():
    global analysis_params
    folder = f'{analysis_params["diretorio"]}/clima/'
    subfolders = [2020, 2021]
    k = 0
    for subfolder in subfolders:
        print(subfolder)
        subfolderpath = f'{folder}{str(subfolder)}'
        for file in os.listdir(subfolderpath):
            print(file.title())
            filepath = f'{subfolderpath}/{file}'
            header = pd.read_csv(
                filepath,
                encoding='latin-1',
                nrows=7,
                parse_dates=True,
                delimiter=';',
                decimal=",",
                header=None
            ).T
            header.columns = header.iloc[0]
            header = header.iloc[1:,1:]
            header = header.rename(columns=lambda x: x.split(':')[0])

            dados_muni_df = pd.read_csv(
                filepath,
                encoding='latin-1',
                skiprows=8,
                parse_dates=True,
                delimiter=';',
                decimal=","
            )
            dados_muni_df['ano'] = pd.to_datetime(dados_muni_df['Data']).dt.year
            dados_muni_df['mes'] = pd.to_datetime(dados_muni_df['Data']).dt.month
            dados_muni_df=dados_muni_df.groupby(by=['ano', 'mes']).agg(
                Precipitacao_mm=('PRECIPITAÇÃO TOTAL, HORÁRIO (mm)', np.sum),
                Pressao_mB=('PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)', np.mean),
                Radiacao_Kj_m2=('RADIACAO GLOBAL (Kj/m²)', np.mean),
                Temp_Celsius=('TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)', np.mean),
                Umidade_Relativa_pct=('UMIDADE RELATIVA DO AR, HORARIA (%)', np.mean),
                Vento_velocidade_m_s=('VENTO, VELOCIDADE HORARIA (m/s)', np.mean)
            ).reset_index()

            for col in header.columns:
                dados_muni_df[col] = header[col].values[0]

            if k==0:
                dados_climaticos_df = dados_muni_df
            else:
                dados_climaticos_df = pd.concat([dados_climaticos_df, dados_muni_df])

            k+=1
    dados_climaticos_df.to_csv(f'{folder}/dados_climaticos.csv')
    return dados_climaticos_df


def prepare_geodataframes_crop(sel_crop_df):
    global analysis_params

    var_selecionada = analysis_params['var_selecionada']

    if sel_crop_df[var_selecionada].count() > 5:

        # Filtra dados ao nivel de municipio
        crop_geodf = sel_crop_df.loc[
            (sel_crop_df[columns_dict['Nivel']] == 4),
            ['geometry', columns_dict['Municipio'], var_selecionada]
        ]

        cols_for_weights = crop_geodf.columns[
            crop_geodf.columns != 'Municipio'
            ]
        crop_geodf = crop_geodf[cols_for_weights]
        crop_full_geodf = crop_geodf.copy(deep=True)
        crop_geodf['old_index'] = crop_geodf.index
        crop_geodf = crop_geodf.dropna().reset_index()
        print('Geodataframe pronto')
    return crop_geodf, crop_full_geodf


# Data summarization
def prepare_summary_harvested_area(crops_df):
    UF_crop_df = pd.pivot_table(crops_df,
                                columns=['Cultura'],
                                index=['Regiao', 'UF_sigla'],
                                values='Area colhida (ha)',
                                fill_value=0,
                                aggfunc=sum
                                ).fillna(0).apply(lambda x:
                                                  round(x / sum(x) * 100, 2)
                                                  ).T
    UF_crop_df = UF_crop_df.dropna()
    # TODO add second subplot that shows total planted area
    f, ax = plt.subplots(figsize=(15, 18))
    sns.heatmap(UF_crop_df,
                annot=True,
                annot_kws={'size': 15},
                linewidths=.5,
                fmt='g',
                ax=ax,
                cbar=False
                )

    ax.tick_params(top=True,
                   labeltop=True,
                   bottom=False,
                   labelbottom=False
                   )

    plt.savefig('heatmap_area_plantada_crop_uf.png')
    plt.close()
    UF_crop_df.to_csv('pivot_table_crop_estado.csv')

    Region_crop_df = pd.pivot_table(crops_df,
                                    columns=['Cultura'],
                                    index=['Regiao'],
                                    values='Area colhida (ha)',
                                    fill_value=0,
                                    aggfunc=sum
                                    ).fillna(0).apply(lambda x:
                                                      round(x / sum(x) * 100, 2)
                                                      ).T
    Region_crop_df = Region_crop_df.dropna()

    # TODO add second subplot that shows total planted area
    f, (ax, ax2) = plt.subplots(nrows=1,
                                ncols=2,
                                sharey=True,
                                figsize=(15, 18),
                                gridspec_kw=dict(width_ratios=[6, 1])
                                )

    sns.heatmap(Region_crop_df,
                annot=True,
                annot_kws={'size': 15},
                linewidths=.5,
                fmt=".0f",
                ax=ax,
                cmap='Blues',
                cbar=False,
                yticklabels=True
                )

    ax.set_xlabel('Região', size=16, position='top')
    ax.set_ylabel('Cultura', size=16)
    ax.tick_params(axis='both',
                   which='major',
                   labelsize=15,
                   top=False,
                   labeltop=True,
                   bottom=False,
                   labelbottom=False
                   )
    ax.xaxis.tick_top()

    total_crop_df = crops_df.groupby('Cultura').agg(
        {'Area colhida (ha)': 'sum'}
    ).dropna()

    total_crop_df['Area colhida (ha)'] = total_crop_df['Area colhida (ha)']/(10 ** 6)
    total_crop_df = total_crop_df.loc[~(total_crop_df == 0).all(axis=1)]

    sns.heatmap(total_crop_df,
                annot=True,
                annot_kws={'size': 15},
                linewidths=.5,
                fmt=".2f",
                ax=ax2,
                cmap=ListedColormap(['white']),
                cbar=False,
                linewidth=1,
                linecolor='black',
                xticklabels=['Area plantada\n ( $\mathregular{ha x 10^{6}}$)']
                )

    ax2.tick_params(top=False, bottom=False,
                    labeltop=True, labelbottom=False)
    ax2.xaxis.set_label_position('top')
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax2.set_yticks([])
    ax2.set_ylabel('')

    plt.tight_layout()
    plt.savefig('heatmap_area_plantada_crop_regiao.png')
    plt.close()

    total_crop_df = total_crop_df.combine_first(Region_crop_df)

    total_crop_df.to_csv('pivot_table_crop_regiao.csv')

    return total_crop_df


def prepare_eda_summary(crops_df):
    table_EDA_crops = crops_df.groupby(['Cultura']).agg(
        {
            'Area plantada (ha)': 'sum',
            'Area colhida (ha)': 'sum',
            'Producao (ton)': 'sum',
            'Produtividade (kg/ha)': 'mean',
            'Receita (Mil Reais)': 'sum'
        }
    )
    convert_to_millions = {
        'Area plantada (ha)': 'Area plantada (Milhoes ha)',
        'Area colhida (ha)': 'Area colhida (Milhoes ha)',
        'Producao (ton)': 'Producao (Milhoes ton)',
        'Receita (Mil Reais)': 'Receita (Milhoes Reais)'
    }

    table_EDA_crops.rename(columns=convert_to_millions, inplace=True)

    for val in convert_to_millions.values():
        if val != 'Receita (Milhoes Reais)':
            table_EDA_crops[val] = table_EDA_crops[val].div(10 ** 6).round(2)
        else:
            table_EDA_crops[val] = table_EDA_crops[val].div(10 ** 3).round(2)

    table_EDA_crops['Produtividade (kg/ha)'] = table_EDA_crops['Produtividade (kg/ha)'].round(1)
    table_EDA_crops.to_csv('tabela_EDA.csv')

    return table_EDA_crops


def prepare_table_yield_classes(sel_crop_df, yield_levels_df):
    global analysis_params

    var_selecionada = analysis_params['var_selecionada']

    niveis_produtividade = ["Muito Baixa",
                            "Baixa",
                            "Media",
                            "Alta",
                            "Muito Alta"
                            ]

    # quantiles_df = sel_crop_df.copy()
    # quantiles_df['class'], intervalos = pd.cut(quantiles_df[var_selecionada], 5, labels=niveis_produtividade, retbins=True)
    q_inf = sel_crop_df[var_selecionada].quantile(0.01)
    q_sup = sel_crop_df[var_selecionada].quantile(0.99)
    remove_outliers = (
            (sel_crop_df[var_selecionada] > q_inf) &
            (sel_crop_df[var_selecionada] < q_sup)
    )

    df_filtered = sel_crop_df.loc[remove_outliers]
    df_filtered['class'], intervalos = pd.cut(
        df_filtered[var_selecionada],
        bins=5,
        labels=niveis_produtividade,
        retbins=True
    )

    intervalos = np.around(np.array(intervalos), decimals=1)

    prod_dict = dict(zip(niveis_produtividade, intervalos))
    totais_regionais_df = df_filtered.groupby(
        ['Cul_7', 'name_regio']
    ).agg(Area_Total=('Are_3', np.sum))

    quantiles_df = df_filtered.groupby(
        ['Cul_7', 'name_regio', 'class']
    ).agg(Area_Faixa_Prod=('Are_3', np.sum))

    quantiles_df = quantiles_df.combine_first(totais_regionais_df).reset_index()
    quantiles_df['Area_Relativa(%)'] = round(
        (quantiles_df['Area_Faixa_Prod'] / quantiles_df['Area_Total']) * 100
        , 1
    )
    quantiles_df['faixa_prod'] = quantiles_df['class'].map(prod_dict)

    if yield_levels_df.empty:
        yield_levels_df = quantiles_df
    else:
        yield_levels_df = pd.concat([yield_levels_df, quantiles_df])
    return yield_levels_df


def prepare_map_crop_quantiles(sel_crop_df):

    global analysis_params

    crop = analysis_params['crop']
    var_nome_original = analysis_params['variavel']
    var_selecionada = analysis_params['var_selecionada']
    weights = analysis_params['weight']
    directory = f'{analysis_params["diretorio"]}/base_dados'
    column_path = f'{directory}/shapes/{crop}/{var_selecionada}'
    # sel_crop_df[var_selecionada] = int(sel_crop_df[var_selecionada])
    missing_kwds = dict(color='grey', label='Sem dados')
    ax = sel_crop_df.plot(column=var_selecionada, legend=True,
                          scheme="quantiles",
                          figsize=(15, 15),
                          legend_kwds=dict(
                              bbox_to_anchor=(0.35, 0.4),
                              fontsize=24,
                            fmt= "{:.0f}"
                          ),
                          missing_kwds=missing_kwds
                          )

    ax.set_axis_off()
    # ax.legend(fontsize=24)
    plt.title('{} \n {}'.format(crop, var_nome_original), fontsize=36)
    plt.savefig(column_path + '/' + crop + '_' + weights + '.png', dpi=200)
    plt.close()


# Spatial analysis
def run_spatial_analysis(crop_geodf):
    global analysis_params
    weights = analysis_params['weight']
    var_selecionada = analysis_params['var_selecionada']

    if weights == "Queen":
        w = lps.weights.Queen.from_dataframe(crop_geodf)
        if w.islands:
            crop_geodf = crop_geodf.drop(w.islands)
            w = lps.weights.Queen.from_dataframe(crop_geodf)
        w.transform = 'r'

        print('Pesos baseados em  movimento Queen preparados')

    elif weights == 'Distance':
        w = lps.weights.DistanceBand.from_dataframe(
            crop_geodf, threshold=30
        )
        if w.islands:
            crop_geodf = crop_geodf.drop(w.islands)
            w = lps.weights.DistanceBand.from_dataframe(
                crop_geodf, threshold=30
            )

        print("Pesos baseados em Distancia preparados")

    crop_geodf[var_selecionada].fillna(0, inplace=True)
    y = crop_geodf[var_selecionada]

    ylag = lps.weights.lag_spatial(w, y)
    mi = esda.moran.Moran(y, w)

    results_crop_df = {
        'Cultura': crop,
        'Municipios': crop_geodf.shape[0],
        "Moran's I": mi.I,
        "p-value": mi.p_norm,
        "z-score": mi.z_norm
    }
    # Brings the latest version of the table
    global table_moran_crop
    
    table_moran_crop = table_moran_crop.append(results_crop_df, ignore_index=True)

    print(crop, '\n',
          var_nome_original, '\n',
          "Moran's I:", mi.I, '\n',
          "Moran's p-value:", mi.p_norm, '\n',
          "Moran's z-score:", mi.z_norm
          )

    return crop_geodf, y, w, mi


def prepare_scatter_plots(moran_loc, mi):

    crop = analysis_params['crop']
    var_nome_original = analysis_params['variavel']
    var_selecionada = analysis_params['var_selecionada']
    weights = analysis_params['weight']
    directory = f'{analysis_params["diretorio"]}/base_dados'
    column_path = f'{directory}/shapes/{crop}/{var_selecionada}'

    # Generates Moran's Global Scatterplot
    fig, ax = plt.subplots(figsize=(12, 12))
    moran_scatterplot(mi, aspect_equal=False, zstandard=True, ax=ax)
    plt.suptitle(f"{crop}, {var_nome_original}"
                 f"\n Moran's p-value: {str(mi.p_norm)}"
                 f"\n Moran's z-score: {str(mi.z_norm)}"
                 )
    plt.savefig(
        f"{column_path}/{crop}_{weights}_moran_scatterplot_global_AC.png",
        bbox_inches='tight'
    )
    plt.close()

    # Generates Moran Local Scatterplot
    fig, ax = moran_scatterplot(moran_loc, p=p)
    ax.set_title("Moran Local Scatterplot", fontsize=22)
    ax.set_xlabel('kg ha\u207b\u00b9', fontsize=18)
    ax.set_ylabel('Spatial Lag', fontsize=18)
    ax.text(1.95, 0.5, "HH", fontsize=25)
    ax.text(1.95, -1, "HL", fontsize=25)
    ax.text(-1, 0.5, "LH", fontsize=25)
    ax.text(-1, -1, "LL", fontsize=25)
    ax.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    plt.savefig(
        column_path + '/' +
        crop + '_' + weights +
        '_' + 'moran_scatterplot_local_AC' + '.png'
    )

    plt.close()


def lisa_cluster_manual(gdf, completo_gdf,Moran_local, HH_only=False):

    global analysis_params
    
    p = analysis_params['p-value']
    weights = analysis_params['weight']
    cultura = analysis_params['crop']
    var_nome_original = analysis_params['variavel']
    fig_path = f'{analysis_params["diretorio"]}/base_dados/shapes/{crop}/{var_selecionada}'
    
    figsize = (15,15)
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
        hmap = colors.ListedColormap(['whitesmoke','lightgrey', 'red', 'lightblue', 'blue', 'pink'])

    gdf = gdf.assign(cl=labels).set_index('old_index')

    completo_gdf = completo_gdf.combine_first(gdf)
    completo_gdf['cl'].fillna('- Sem dados', inplace=True)
    ax = completo_gdf.plot(column='cl', categorical=True, \
                                    k=2, cmap=hmap, linewidth=0.1, ax=ax, \
                                    edgecolor='white', legend=True, legend_kwds=dict(
                              bbox_to_anchor=(0.25, 0.4),
                              fontsize=24
                          ))
    ax.set_axis_off()
    # ax.legend(fontsize=24)
    plt.title('{} \n {} \n {}'.format(cultura, var_nome_original, 'Mapa de Hot e Cold spots'), fontsize=36)

    f.tight_layout()

    # Display the figure
    plt.savefig(f'{fig_path}/{cultura}_{weights}_LISA_clusters.png',
                bbox_inches='tight'
                )
    plt.close()


directory = f'{analysis_params["diretorio"]}/base_dados'
climate_directory = f'{analysis_params["diretorio"]}/clima'
soil_directory = f'{analysis_params["diretorio"]}/solos'

if not read_csv:
    crops_df = prepare_crops_dataframe()
    climate_df = prepare_climate_data()

else:
    crops_df = pd.read_csv(directory + '/crops.csv')
    
    climate_df = pd.read_csv(
        climate_directory+'/dados_climaticos.csv', decimal=','
    )

soils_gdf = gpd.read_file(f'{soil_directory}/Solos_5000.shp')
climate_koppen_gdf = gpd.read_file(f'{climate_directory}/CLK_21_BR_CEM.shp')

crops_df = crops_df.loc[(crops_df['Nivel'] == 4) & (crops_df['Cultura'] != 'Total')
            ]

climate_gdf = gpd.GeoDataFrame(
    climate_df,
    geometry=gpd.points_from_xy(climate_df['LONGITUDE'], climate_df['LATITUDE']
                                )
)

estados_dict, regioes_dict = prepare_dict_estados_regioes()

estados = list(estados_dict.keys())

crops_df = format_crops_dataframe(crops_df=crops_df)


if prepare_summary:
    table_eda_summary = prepare_eda_summary(crops_df=crops_df)
    area_colhida_df = prepare_summary_harvested_area(crops_df=crops_df)


columns_dict = {column: column[:3]+ '_' + str(k)
                for k, column in enumerate(crops_df.columns)
                }
crops_df.columns = [value for value in columns_dict.values()]


# Coluna definida para analise espacial

var_selecionada = columns_dict[var_nome_original]

analysis_params['var_selecionada'] = var_selecionada

shape_path = directory + '/shapes'

if not read_shape:
    # geodata = geobr.read_municipality(code_muni="all", year=2019)
    geodata = gpd.read_file(directory+"/geodata.shp")

if not crops:
    crops = set(crops_df[columns_dict['Cultura']])

# Filtra e remove culturas sem resultados para a variavel de interesse
crops_nao_nulas = crops_df.groupby(
    [columns_dict['Cultura']]
)[var_selecionada].sum().loc[lambda x : x > 0].index.tolist()

crops_minimo_cidades = crops_df.groupby(
    [columns_dict['Cultura']]
)[var_selecionada].count().loc[lambda x : x > 30].index.tolist()

crops = [value for value in crops if value in crops_nao_nulas]

crops = sorted([value for value in crops if value in crops_minimo_cidades])

table_moran_crop = pd.DataFrame(
    columns=['Cultura', 'Municipios', "Moran's I", "p-value", "z-score"]
)

n = 0

yield_levels_df = pd.DataFrame()

for crop in crops:
    print(f'Preparing: {crop} \n '
          f'crop {n+1} of {len(crops)}')
    analysis_params['crop'] = crop
    shape_path = f'{analysis_params["diretorio"]}/base_dados/shapes/{crop}'
    shape_filename = shape_path + '/' + crop + '.shp'
    if read_shape:
        sel_crop_df = gpd.read_file(shape_filename)
    else:
        sel_crop_df = crops_df.loc[
            (crops_df[columns_dict['Cultura']] == crop) &
            (crops_df[columns_dict['Nivel']] == 4)
            ]
        IQ1, _, IQ3=sel_crop_df[var_selecionada].quantile([0.25, 0.5, 0.75])
        IQR = IQ3 - IQ1
        lower_bound= IQ1 - (1.5 * IQR)
        upper_bound = IQ3 + (1.5 * IQR)
        # filtered_range = (sel_crop_df[var_selecionada]>=lower_bound) & (sel_crop_df[var_selecionada]<=upper_bound)

        # sel_crop_df = sel_crop_df[filtered_range]
        prod_values = sel_crop_df[var_selecionada].copy().values
        prod_values = np.where((prod_values < lower_bound), np.nan, prod_values)
        prod_values = np.where((prod_values > upper_bound), np.nan, prod_values)# sel_crop_df = sel_crop_df.loc[sel_crop_df['Municipio'].notnull()]
        sel_crop_df[var_selecionada] = prod_values
        # geodata.to_csv(directory+'/geodata.csv')
        # Padronizando regras de nomenclatura entre os dois datasets
        geodata.loc[:, ('name_muni')] = \
            geodata.loc[:,('name_muni')].str.lower().apply(
                unidecode).str.replace('-', ' ')
        copy_df = sel_crop_df.copy()
        mask = columns_dict['Municipio']

        copy_df.loc[:,(mask)] = copy_df.loc[:, mask].str.lower().replace('-', ' ').apply(unidecode)

        sel_crop_df = copy_df
        sel_crop_df = pd.merge(
            geodata, sel_crop_df,
            right_on=[columns_dict['UF_sigla'], columns_dict['Municipio']],
            left_on=['abbrev_sta', 'name_muni'],
            how='outer'
        )

        sel_crop_df = gpd.GeoDataFrame(sel_crop_df)

        if not os.path.exists(shape_path):
            os.makedirs(shape_path)
        sel_crop_df.to_file(shape_filename)


    crs_value = sel_crop_df.crs
    print(crs_value)
    cultura_climate_gdf = gpd.sjoin(sel_crop_df, climate_gdf.set_crs(crs=crs_value))

    # Classifica a area colhida em cinco niveis de produtividade
    # TODO verify if indentation level is correct
    yield_levels_df = prepare_table_yield_classes(
        sel_crop_df=sel_crop_df,
        yield_levels_df=yield_levels_df
    )

    column_path = shape_path + "/" + str(var_selecionada)
    if not os.path.exists(column_path):
        os.makedirs(column_path)

    # TODO quantiles must disregard outliers and be done in terms of (Area or Yield)
    # Prepares the maps for crop with yield quantiles
    prepare_map_crop_quantiles(sel_crop_df)

    crop_geodf, crop_full_geodf = prepare_geodataframes_crop(sel_crop_df)

    crop_geodf, y, w, mi = run_spatial_analysis(crop_geodf=crop_geodf)

    moran_loc = Moran_Local(y, w)

    # Prepares local and global scatter plots
    prepare_scatter_plots(moran_loc, mi)

    lisa_cluster_manual(gdf=crop_geodf,
                        completo_gdf=crop_full_geodf,
                        Moran_local=moran_loc
                        )

    n += 1
table_moran_crop['p-valor'] = ['Significativo' if p<0.05 else 'Não Significativo' for p in table_moran_crop['p-value']]
table_moran_crop.to_csv(directory+"/summary_table_crops.csv")

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10))
p1 = sns.scatterplot(table_moran_crop, x='Municipios', y="Moran's I", hue='p-valor', ax=ax)

for line in range(0,table_moran_crop.shape[0]):
    if table_moran_crop["Municipios"][line] > 2500:
         p1.text(table_moran_crop["Municipios"][line]+0.01, table_moran_crop["Moran's I"][line],
         table_moran_crop["Cultura"][line], horizontalalignment='left',
         size='medium', color='black', weight='semibold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('Municipios', fontsize=18)
plt.ylabel('Estátistica Global de Moran (I)', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig(directory+"/global_moran_crops.png")
plt.close()

yield_levels_df.to_csv(directory+"/yield_levels.csv")
region_level_df = yield_levels_df.groupby(['name_regio', 'class']).agg(Total=('Area_Faixa_Prod', np.sum)).reset_index()
region_level_df['Milhoes ha'] = region_level_df['Total']/(10**6)

sns.set(font_scale=2.5)
g = sns.catplot(region_level_df,
                x='class',
                y='Milhoes ha',
               palette='Blues',
                row='name_regio',
                row_order=['Norte', 'Nordeste', 'Centro Oeste', 'Sudeste', 'Sul'],
                kind='bar',
                sharey=False,
                height=6, aspect=1.5,
                order=["Muito Baixa","Baixa","Media","Alta","Muito Alta"]
                )
# iterate through axes
for ax in g.axes.ravel():

    # add annotations
    for c in ax.containers:
        labels = [f'{(v.get_height()):.1f}' for v in c]
        ax.bar_label(c, labels=labels, label_type='edge')
    ax.margins(y=0.2)



g.set_ylabels("ha x $10^6$")
g.set_titles(template='{row_name}', fontdict={'fontsize': 20},y=0.90, x=-0.1, ha='right')
plt.tight_layout()
plt.savefig(directory+"/regions_yield_classes.png")