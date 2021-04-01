import pandas as pd
import numpy as np
from pandas import DataFrame
import statistics, dedupe, json, os, csv, re, unidecode, urllib.parse, requests
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import imblearn 
from imblearn import under_sampling
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier  

plt.style.use('ggplot')

# readData comes from dedup tutorial
def readData(filename):
    """
    Read in data from a CSV file and create a dictionary of records,
    where the key is a unique record ID.
    """

    data_d = {}

    with open(filename) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            clean_row = dict([(k, preProcess(v)) for (k, v) in row.items()])
            data_d[i] = dict(clean_row)

    return data_d

def preProcess(column):
    """
    Do a little bit of data cleaning with the help of Unidecode and Regex.
    Things like casing, extra spaces, quotes and new lines can be ignored.
    """

    column = unidecode.unidecode(column)
    column = re.sub('\n', ' ', column)
    column = re.sub('/', ' ', column)
    column = re.sub("'", '', column)
    column = re.sub(":", ' ', column)
    column = re.sub('  +', ' ', column)
    column = column.strip().strip('"').strip("'").upper().strip()
    if not column:
        column = None
    return column

# clusterMerge adapted from dedupe tutorial
def clusterMerge(datasource, key, values):
    clusters = datasource[datasource[values].notnull()]
    to_cluster = {}
    for i,row in clusters.iterrows():
        dictKey = row[key]
        dictValues = row[values]
        if dictKey not in to_cluster:
            to_cluster[dictKey] = [dictValues]
        else:
            to_cluster[dictKey].append(dictValues)
    return to_cluster

def load_data(path_to_spreadsheet, columns, label, rename_columns=None, column_edits=None):
    '''
    This function loads in the data that needs to be processed.

    path_to_spreadsheet: string pointing to file
    columns: list of columns to retain
    label: label high or unknown risk (1 for high risk, 0 for unknown)
    rename_columns: dictionary in form of {'current name': 'new name'}
    column_edits: list of lists. [[column, find value, replace value]]
    '''
    data = pd.read_csv(path_to_spreadsheet)
    data_subset = data[columns]
    data_subset['labels'] = label
    if rename_columns:
        data_subset.rename(columns=rename_columns, inplace=True)
    if column_edits:
        for column_edit in column_edits:
            data_subset[column_edit[0]] = data_subset[column_edit[0]].str.replace(column_edit[1], column_edit[2]) 

    return data_subset

# information to import
files_to_import = [
    ('zaubacorp_garments_Initial_NIC_Set.csv', ['company_name', 'cin', 'address', 'authorised_capital', 'paid_up_capital', 'email'], 0),
    ('zaubacorp_textile_Initial_NIC_Set.csv', ['company_name', 'cin', 'address', 'authorised_capital', 'paid_up_capital', 'email'], 0),
    ('all_imports__garments_shipments_results_1_to_10_000_of_25_507.csv', ['Consignee', 'Consignee Full Address'], 0),
    ('all_imports__garments_shipments_results_1_to_10_000_of_25_507.csv', ['company_name', 'address', 'Shipper Email 1'], 0, {'Shipper Email 1':'email'}),
    ('Apparel_Industry_ Directly_connected_companies.csv', ['company_name', 'cin', 'address'], 1),
    ('Apparel_Industry_Companies_connected_to_violators.csv', ['company_name', 'cin', 'address'], 1),
    ('Apparel_Industry_Companies_connected_to_violators.csv', ['suspected violator'], 1, {'suspected violator':'company_name'}),
    ('badlist.csv', ['company_name', 'address'], 1),
    ('textile_Sector_Violators_Directly_sourced_matches.csv', ['company'], 1, {'company':'company_name'}),
    ('textile_Sector_Violators_Directly_sourced_matches.csv', ['group'], 1, {'group':'company_name'}),
    ('Marks and Spencers Data Match and Close.csv',['company_name', 'address', 'Total workers', r'% Female', 'Trade Union', 'Workers Commitee'], 0),
    ('facilities.csv', ['company_name', 'address', 'lat', 'lng','contributors'], 0, {'contributors':'Consignee'},[['Consignee', r'\([^)]*\)', '']])
]

# load the data
companies = pd.concat([load_data(*import_file) for import_file in files_to_import])


#Clean up the strings, put them in upper case, remove punctuation from names, remove low-info parts of strings
upper_columns = ['company_name', 'address', 'Consignee']
for upper_column in upper_columns:
    companies[upper_column] = companies[upper_column].str.upper()

strip_columns = ['company_name']
for strip_column in strip_columns:
    companies[strip_column] = companies[strip_column].str.strip()

replace_columns = [
    ['company_name', '.', ''],
    ['Consignee', r'\s*(LTD PVT|LIMITED PRIVATE|LTD|PVT|INC|LIMITED|PRIVATE|LTD PRIVATE|LIMITED PVT|PVT LTD|PRIVATE LIMITED|PRIVATE LTD|PVT LIMITED)\s*$',''],
    ['authorised_capital','₹',''],
    ['authorised_capital',',',''],
    ['paid_up_capital','₹',''],
    ['paid_up_capital',',','']
]
for replace_column in replace_columns:
    companies[replace_column[0]] = companies[replace_column[0]].str.replace(replace_column[1], replace_column[2])

# save interim state to file for future reference
companies.to_csv("alldata.csv")


input_file = "alldata.csv"
output_file = 'canonical_businesses.csv'
settings_file = 'business_dedupe_learned_settings_new'
training_file = 'csv_dedupe_training_new.json'

# the following code is based on the dedup tutorial

print('importing data ...')
data_d = readData(input_file)


if os.path.exists(settings_file):
    print('reading from', settings_file)
    with open(settings_file, 'rb') as f:
        deduper = dedupe.StaticDedupe(f)
else:

    fields = [
        {'field': 'company_name', 'type': 'String', 'has missing': True},
        {'field': 'address', 'type': 'String', 'has missing': True},
        {'field': 'cin', 'type':'String', 'has missing': True},
        {'field': 'email', 'type': 'String', 'has missing': True}
        ]
    
    deduper = dedupe.Dedupe(fields)

    if os.path.exists(training_file):
        print('reading labeled examples from ', training_file)
        with open(training_file, 'rb') as f:
            deduper.prepare_training(data_d, f)
    else:
        deduper.prepare_training(data_d)

    #Active labeling
    print('starting active labeling...')

    dedupe.console_label(deduper)

    deduper.train()

    with open(training_file, 'w') as tf:
        deduper.write_training(tf)

    with open(settings_file, 'wb') as sf:
        deduper.write_settings(sf)
    
    print('clustering...')

clustered_dupes = deduper.partition(data_d, 0.3)

print('# duplicate sets', len(clustered_dupes))

#write to file with cluster IDs
companies['cluster'] = 0
companies['confidence'] = 0
cluster_membership = {}
for cluster_id, (records, scores) in enumerate(clustered_dupes):
    for record_id, score in zip(records, scores):
        record_id = int(record_id)
        companies['cluster'].iloc[record_id] = cluster_id
        companies['confidence'].iloc[record_id] = score
        # cluster_membership[record_id] = {
        #     "Cluster ID": cluster_id,
        #     "confidence_score": score
        # }

# assign a unique id to each company
companies['uid'] = range(len(companies))

# save dataframe with cluster info included
companies.to_csv("clustereddata.csv")


# #companies['labels'] = companies['labels'].fillna(0)
# #Start building clusters
# cluster_to_name = clusterMerge(companies,'cluster','company_name')
# cluster_to_address = clusterMerge(companies, 'cluster', 'address')
# name_to_cluster = clusterMerge(companies,'company_name','cluster')
# address_to_cluster = clusterMerge(companies, 'address', 'cluster')
# #I think lines related to email need to be added here

# One useful characteristic is how many trade partners an entity has
tradePartners = {}
companies['trade_partner_count'] = 0
for i,row in companies.iterrows():
    current_company = row['cluster'] # grab the cluster id to refer to deduped company
    if type(row['Consignee']) != float: # sometimes Consignees are floats (messy data)
        partners = row['Consignee'].split("|") # partners are divided bypipes

        if current_company not in tradePartners: # if we have not seen the comapny before
            tradePartners[current_company] = set(partners)  # save a set of the partners
        else:
            tradePartners[current_company].union(partners) # otherwise, add them to the set

# prep a dictionary to store the new information (we will turn this into a dataframe later),
# based on cluster information
refined_data = {}

column_names = list(companies.columns)
for i,row in companies.iterrows():
    # if we've seen this cluster, count how many trade partners it has
    if row['cluster'] in tradePartners:
        row['trade_partner_count'] = len(tradePartners[row['cluster']])
    
    # if we have not encountered the cluster yet transfer the information from the row to the dictionary
    if row['cluster'] not in refined_data:
        refined_data[row['cluster']] = dict(row)
    else:
        for column in column_names:
            stored_val = refined_data[row['cluster']][column]
            if stored_val == "":
                refined_data[row['cluster']][column] = row[column]
            if column == "labels":
                # if the current value is 1 replace stored value
                if row[column] == 1:
                    refined_data[row['cluster']][column] = row[column]
            if column == "address":
                # grab the longest address
                if type(refined_data[row['cluster']][column]) == str and type(row[column]) == str:
                    if len(row[column]) > len(refined_data[row['cluster']][column]):
                        refined_data[row['cluster']][column] = row[column]

# Create a data frame out of the refined data
dict_list = [data_dict for data_dict in refined_data.values()]
data_add_geo = pd.DataFrame(dict_list,columns=column_names)

# Save interim data to file
data_add_geo.to_csv("deduped_data.csv")

# Get location information. If nothing has been saved, run google api. Otherwise grab files from address folder
# google maps api
key = 'redacted' 
apiURL = "https://maps.googleapis.com/maps/api/geocode/json?address="
have_address = data_add_geo.dropna(subset=['address'])
address_directory = "addressresults"
if not os.path.isdir(address_directory):
    os.mkdir(address_directory)
# get lat/lon for addressed where the info is not available yet
for i,row in have_address.iterrows():
    address = row['address']
    lat = row['lat']
    lon = row['lng']
    if np.isnan(lat) and np.isnan(lon):
        row_id = str(row['uid'])
        fname = row_id +".json" 
        if not os.path.isfile(os.path.join(address_directory,fname)): 
            # use google maps api to fetch lat/lon
            address = urllib.parse.quote(address, safe='')
            apiCall = f'{apiURL}{address}&key={key}'
            print(f'Calling api for {fname}')
            results = requests.get(apiCall)
            results_json = results.json()
            if "results" in results_json and len(results_json["results"]) > 0:
                result = results_json["results"][0]
                if "geometry" in result:
                    lat = result["geometry"]["location"]["lat"]
                    lon = result["geometry"]["location"]["lng"]
            with open(os.path.join(address_directory,fname), 'w', encoding='utf8') as wf:
                json.dump(results_json, wf)

        else:
            # load lat long from file
            with open(os.path.join(address_directory,fname), 'r', encoding='utf8') as rf:
                results_json = json.load(rf)
            if "results" in results_json and len(results_json["results"]) > 0:
                result = results_json["results"]
                if "geometry" in result:
                    lat = result["geometry"]["location"]["lat"]
                    lon = result["geometry"]["location"]["lng"]

        # save the info back to the dataframe
        data_add_geo.loc[i, "lat"] = lat
        data_add_geo.loc[i, "lng"] = lon



training_frame = data_add_geo.dropna(subset=['company_name','address','authorised_capital','paid_up_capital','trade_partner_count','labels','lat','lng'])

# columns to use in training/test, can be adjusted as you please:
training_columns = ["authorised_capital", "paid_up_capital", "trade_partner_count", "lat", "lng"]

#Downsample
training_frame_majority = training_frame[training_frame.labels == 0]
training_frame_minority = training_frame[training_frame.labels == 1]
# training_frame_minority.to_csv("highrisk.csv")

training_frame_majority_downsampled = training_frame_majority.sample(n=100, random_state=42)
# training_frame_majority_downsampled.to_csv("controlgroup.csv")

training_frame_downsampled = pd.concat([training_frame_majority_downsampled, training_frame_minority])

#Create numpy arrays for features and target
plotdata = training_frame_downsampled[training_columns]
X = training_frame_downsampled[training_columns].values
y = training_frame_downsampled['labels'].values

#importing train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42, stratify=y)

#Rescaling the features

scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

#Import the classifier

classifier = KNeighborsClassifier(n_neighbors=1, weights = 'distance')  
classifier.fit(X_train, y_train)

#pickle the model
#import pickle
#pickle.dump((classifier, scaler), open('rfendstate.p','wb'))

#Make the prediction
y_pred = classifier.predict(X_test)  

#Plot the results using t_SNE


np.random.seed(42)
rndperm = np.random.permutation(plotdata.shape[0])

N = 80

df_merge = pd.DataFrame(data=X_train, columns=training_columns)

df_merge = df_merge[:N]

'''

'''
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=250)
# tsne_results = tsne.fit_transform(df_merge.values)
# df_merge["label"] = y_train[:N]
# df_merge['tsne-2d-one'] = tsne_results[:,0]
# df_merge['tsne-2d-two'] = tsne_results[:,1]
# plt.figure(figsize=(16,10))
# sns.set_style("white")
# sns.scatterplot(
#     x="tsne-2d-one", y="tsne-2d-two",
#     hue="label",
#     palette=sns.color_palette("dark", 2),
#     data=df_merge,
#     #legend='full',
#     alpha=0.3
# )
# legend = plt.legend()
# legend.get_texts()[1].set_text('Low Risk')
# legend.get_texts()[2].set_text('High Risk')
# sns.despine()
# plt.savefig("RFPlot.png")
#plt.show()

#Print the output if you want. Otherwise write it to file.
#print(y_pred)

feature_imp = pd.Series(classifier.feature_importances_,index=training_frame[training_columns].columns).sort_values(ascending=False)
print(feature_imp)
'''
'''
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))  
