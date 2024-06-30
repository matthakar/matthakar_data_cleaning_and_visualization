import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import textwrap
sns.set()

# define file paths

input_directory = r'C:\Users\matth\Documents\Matt Hakar\Python\Github Scripts\matthakar_data_cleaning_and_visualization'
collated_df_path = r'C:\Users\matth\Documents\Matt Hakar\Python\Github Scripts\matthakar_data_cleaning_and_visualization\collated_df.csv'
shoes_df_path = r'C:\Users\matth\Documents\Matt Hakar\Python\Github Scripts\matthakar_data_cleaning_and_visualization\shoes_df.csv'

'''
The purpose of this Python script is to pull Amazon footwear price information for the top-rated shoes (>= 4 stars) and visualize some of the data.

The dataset used for this script can be found here: https://www.kaggle.com/datasets/lokeshparab/amazon-products-dataset?resource=download

This script is split into three separate parts. It can be run at once or in separate segments when modifying the data analysis/visualization outputs to save on runtime.

Part 1. Collate Amazon product files into one df
Part 2. Clean and reformat the collated_df to create the shoes_df
Part 3. Further clean data and visualize price data from the shoes_df
'''
#####################################################################################################

'''
Part 1. Collate Amazon product files into one df
'''

collated_df = pd.DataFrame()

for file in os.listdir(input_directory):
    file_name = os.fsdecode(file)
    path_name = input_directory + '/' + file_name
    if path_name.endswith('.csv') and file_name != 'collated_df.csv' and file_name != 'shoes_df.csv':
        input_df = pd.read_csv(path_name)
        collated_df = pd.concat([collated_df, input_df], axis=0)

collated_df.to_csv(collated_df_path, index=False)

'''
Part 2. Clean and reformat the collated_df to create the shoes_df
'''

df = pd.read_csv(collated_df_path)

df = df.iloc[:, 0:9]

# check the currency for the price columns

def contains_rupee_symbol(column):
    return column.apply(lambda x: '₹' in str(x)).any()

# filter out rows that don't contain the rupee symbol

df = df.dropna(subset=['discount_price', 'actual_price'])

filt_1 = df['discount_price'].str.contains('₹')
filt_2 = df['actual_price'].str.contains('₹')

df = df.loc[filt_1,:]
df = df.loc[filt_2,:]

# remove image and link columns from df

columns_to_remove = ['image', 'link']
df = df.drop(columns=columns_to_remove)

# remove commas and rupee symbols from price columns

price_columns = ['discount_price', 'actual_price']

df[price_columns] = df[price_columns].replace({'₹': '', ',': ''}, regex=True)

# print(df.columns)

df = df.rename(columns={'name':'Product Name', 'main_category':'Category', 'sub_category':'Subcategory', 'ratings':'Product Rating', 'no_of_ratings':'Rating Count', 'discount_price':'Discount Price (INR)', 'actual_price':'Actual Price (INR)'})

# replace rupee price columns (INR) with US dollar price columns (USD)

rupee_to_usd_conversion_rate = 0.012  # 1 INR = 0.012 USD

def convert_to_usd(rupee_value, rate):
    numeric_value = float(rupee_value)
    return numeric_value * rate

df['Discount Price (USD)'] = df['Discount Price (INR)'].apply(lambda x: convert_to_usd(x, rupee_to_usd_conversion_rate))
df['Actual Price (USD)'] = df['Actual Price (INR)'].apply(lambda x: convert_to_usd(x, rupee_to_usd_conversion_rate))

rupee_price_columns = ['Discount Price (INR)', 'Actual Price (INR)']
df = df.drop(columns=rupee_price_columns)

#create shoes_df

filt = (df['Category'] == "women's shoes") | (df['Category'] == "men's shoes")

shoes_df = df.loc[filt,:]

unique_categories = shoes_df['Subcategory'].unique()

# convert rating columns to numeric values and drop rows without a rating

shoes_df['Product Rating'] = pd.to_numeric(shoes_df['Product Rating'], downcast='float', errors='coerce')
shoes_df['Rating Count'] = pd.to_numeric(shoes_df['Rating Count'], downcast='integer', errors='coerce')

shoes_df = shoes_df.dropna(subset=['Product Rating', 'Rating Count'])

filt = (shoes_df['Product Rating'] >= 4)

shoes_df = shoes_df.loc[filt].copy()

# round USD prices to the nearest cent

price_columns = ['Discount Price (USD)', 'Actual Price (USD)']

shoes_df[price_columns] = shoes_df[price_columns].round(2)

shoes_df.to_csv(shoes_df_path, index=False)

'''
Part 3. Further clean footwear data and visualize price data from the shoes_df
'''

df = pd.read_csv(shoes_df_path)

filt1 = df['Category'] == "men's shoes"
filt2 = df['Category'] == "women's shoes"
base_men_shoes_df = df.loc[filt1]
base_women_shoes_df = df.loc[filt2]

# identify and remove miscategorized shoes from the each dataframe

men_unique_categories = base_men_shoes_df['Product Name'].unique()
women_unique_categories = base_women_shoes_df['Product Name'].unique()

miscategorized_men_shoes_list = [category for category in men_unique_categories if ('women' in category.lower() or 'woman' in category.lower()) and ' men' not in category.lower()]
miscategorized_women_shoes_list = [category for category in women_unique_categories if (' men' in category.lower() or ' man' in category.lower()) and ' women' not in category.lower()]

# print(miscategorized_men_shoes_list)
# print(miscategorized_women_shoes_list)

string = '|'.join([f'^{item}$' for item in miscategorized_men_shoes_list])
string = re.escape(string)
base_men_shoes_df = base_men_shoes_df[~base_men_shoes_df['Product Name'].str.contains(string, case=False)]

string = '|'.join([f'^{item}$' for item in miscategorized_women_shoes_list])
string = re.escape(string)
base_women_shoes_df = base_women_shoes_df[~base_women_shoes_df['Product Name'].str.contains(string, case=False)]

'''
test with one of the miclassified shoes to make sure it was filtered out correctly, df comes back as Empty
'''
# filt3 = base_men_shoes_df['Product Name'] == 'Skechers Womens Go Walk 5 - Downdraft Walking Shoes'
# base_men_shoes_test = base_men_shoes_df.loc[filt3]
# print(base_men_shoes_test)

# remove duplicate entries with both the same product name, actual, and discount price

base_men_shoes_df = base_men_shoes_df.drop_duplicates(subset=['Product Name', 'Actual Price (USD)', 'Discount Price (USD)'], keep='first')
base_women_shoes_df = base_women_shoes_df.drop_duplicates(subset=['Product Name', 'Actual Price (USD)', 'Discount Price (USD)'], keep='first')

def calculate_avg_and_std(df, category_column, value_column):
    unique_categories = df[category_column].unique()
    output_dict = {}
    for category in unique_categories:
        values = df.loc[df[category_column] == category, value_column]
        avg_value = values.mean()
        std_dev = values.std()
        output_dict[category] = (avg_value, std_dev)
    return output_dict

women_shoe_price_stats = calculate_avg_and_std(base_women_shoes_df, 'Subcategory', 'Actual Price (USD)')
# print(women_shoe_price_stats)
women_shoe_discount_stats = calculate_avg_and_std(base_women_shoes_df, 'Subcategory', 'Discount Price (USD)')
# print(women_shoe_discount_stats)
men_shoe_price_stats = calculate_avg_and_std(base_men_shoes_df, 'Subcategory', 'Actual Price (USD)')
# print(men_shoe_price_stats)
men_shoe_discount_stats = calculate_avg_and_std(base_men_shoes_df, 'Subcategory', 'Discount Price (USD)')
# print(men_shoe_discount_stats)

# convert stat dictionaries to dataframes

def convert_dict_to_df(dictionary):
    key_list = list(dictionary.keys())
    average_list = [value[0] for value in dictionary.values()]
    std_dev_list = [value[1] for value in dictionary.values()]
    key_df = pd.DataFrame(key_list, columns=['Subcategory'])
    average_df = pd.DataFrame(average_list, columns=['Average Price (USD)'])
    std_dev_df = pd.DataFrame(std_dev_list, columns=['Standard Deviation (USD)'])
    df = pd.concat([key_df, average_df, std_dev_df], axis=1)
    return df

women_shoe_price_df = convert_dict_to_df(women_shoe_price_stats)
women_shoe_discount_df = convert_dict_to_df(women_shoe_discount_stats)
men_shoe_price_df = convert_dict_to_df(men_shoe_price_stats)
men_shoe_discount_df = convert_dict_to_df(men_shoe_discount_stats)

# make initial bar graphs to visualize the averages for women and men footwear for both actual and discount prices

def average_price_bar_graph(df, df_type, price_type):
    if df_type == 'women':
        palette_dict = {
        'Ballerinas': '#ff6f52',
        'Fashion Sandals': '#069af3',
        'Shoes': '#ffdf22'}
    elif df_type == 'men':
        palette_dict = {
        'Casual Shoes': '#ff6f52',
        'Formal Shoes': '#069af3',
        'Sports Shoes': '#ffdf22'}
    plt.figure(figsize= (10.95,4.23))
    sns.set_style('whitegrid')
    plot = sns.barplot(data=df, x= 'Subcategory', y= 'Average Price (USD)', width=1, hue='Subcategory', palette=palette_dict, dodge=False, hue_order=list(palette_dict.keys()))
    plt.subplots_adjust(top=0.82, bottom=0.18, left=0.083, right=0.93)  
    plt.margins(x=0, y=0)
    plt.xticks(rotation=0, horizontalalignment='center', fontweight='bold', fontsize=14)
    plt.yticks([0,10,20,30,40,50],fontsize = 14, fontfamily='Calibri',fontweight='bold')
    plt.ylim(0,50)
    if df_type == 'men' and price_type == 'actual':
        plt.title("Average Top-Rated Men's Footwear Price by Category", y=1.05, fontsize=21.9, fontfamily='Calibri', fontweight='bold', pad=15)
    elif df_type == 'men' and price_type == 'discount':
        plt.title("Average Top-Rated Men's Footwear Discount Price by Category", y=1.05, fontsize=21.9, fontfamily='Calibri', fontweight='bold', pad=15)
    elif df_type == 'women'and price_type == 'actual':
        plt.title("Average Top-Rated Women's Footwear Price by Category", y=1.05, fontsize=21.9, fontfamily='Calibri', fontweight='bold', pad=15)
    elif df_type == 'women' and price_type == 'discount':
        plt.title("Average Top-Rated Women's Footwear Discount Price by Category", y=1.05, fontsize=21.9, fontfamily='Calibri', fontweight='bold', pad=15)      
    plt.xlabel('Type of Footwear', fontsize=18, fontfamily='Calibri',fontweight='bold', labelpad=13)
    plt.ylabel('Average Price (USD)', x=0,fontsize=18, fontfamily='Calibri',fontweight='bold', labelpad=13)
    plt.show()

average_price_bar_graph(women_shoe_price_df, 'women', 'actual')
average_price_bar_graph(women_shoe_discount_df, 'women', 'discount')
average_price_bar_graph(men_shoe_price_df, 'men', 'actual')
average_price_bar_graph(men_shoe_discount_df, 'men', 'discount')

# make bar graphs with all women data on one bar graph and all men data on one bar graph

women_shoe_df = pd.concat([women_shoe_price_df, women_shoe_discount_df], axis=1)
men_shoe_df = pd.concat([men_shoe_price_df, men_shoe_discount_df], axis=1)

new_names = {1:'Actual', 4: 'Discount'}

def rename_columns_by_index(df):
    new_columns = [new_names.get(i, col) for i, col in enumerate(df.columns)]
    df.columns = new_columns

rename_columns_by_index(women_shoe_df)
rename_columns_by_index(men_shoe_df)

women_shoe_df = women_shoe_df.iloc[:,[0,1,4]]
# print(women_shoe_df)
men_shoe_df = men_shoe_df.iloc[:,[0,1,4]]
# print(men_shoe_df)

def combined_bar_graph(df, df_type):
    df_melted = pd.melt(df, id_vars=['Subcategory'], value_vars=['Actual', 'Discount'], var_name='Metric', value_name='Average Price (USD)')
    palette_colors = ['#ff6f52', '#069af3']
    plt.figure(figsize=(10.95,4.23))
    sns.set_style('whitegrid')
    plot = sns.barplot(x='Subcategory', y='Average Price (USD)', hue='Metric', palette=palette_colors, data=df_melted)
    plt.subplots_adjust(top=0.82, bottom=0.18, left=0.083, right=0.871)
    plt.margins(x=0, y=0)
    plt.xticks(rotation=0, horizontalalignment='center', fontweight='bold', fontsize=14)
    plt.yticks([0,10,20,30,40,50],fontsize=14, fontfamily='Calibri',fontweight='bold')
    plt.ylim(0,50)
    plot.legend(title='Price', loc='center left', bbox_to_anchor=(1, 0.5))
    if df_type == 'men':
        plt.title("Average Top-Rated Men's Footwear Price by Category", y=1.05, fontsize = 23, fontfamily='Calibri', fontweight= 'bold', pad=15)
    elif df_type == 'women':
        plt.title("Average Top-Rated Women's Footwear Price by Category", y=1.05, fontsize = 23, fontfamily='Calibri', fontweight= 'bold', pad=15)      
    plt.xlabel('Type of Footwear', fontsize=18, fontfamily='Calibri',fontweight='bold', labelpad=13)
    plt.ylabel('Average Price (USD)', x=0,fontsize=18, fontfamily='Calibri',fontweight='bold', labelpad=13)
    plt.show()
 
combined_bar_graph(women_shoe_df, 'women')
combined_bar_graph(men_shoe_df, 'men')

'''
Given the high standard deviations for these top-rated footwear options, it makes sense to see how these prices are distributed for a smaller subsection of the data
'''

# men's casual shoe analysis

filt = base_men_shoes_df['Subcategory'] == 'Casual Shoes'

casual_df = base_men_shoes_df.loc[filt]
casual_df = casual_df.iloc[:, [0,6]]

casual_df = casual_df.sort_values(by=casual_df.columns[1], ascending=False)

# specify only the running shoes

nike_running_df = casual_df[casual_df['Product Name'].str.contains('Nike', case=False, na=False) & casual_df['Product Name'].str.contains('running', case=False, na=False)]

# clean the running shoe names so that they aren't as long - important for visualization on the bar graph

replacement_dict = {'Nike ':'', ' Running Shoes':'', ' Running Shoe':'', 'Men ':'', 'Mens ':'', 'mens ': '', "Men's ": '', 'Running': '', ' Black/Hyper Blue (CD7093-011':'', 'Boys ':'', 'Black/Hyper Blue (CD7093-011)':'', ', Black, 6 US, Black/White (AT4209)':'', ', 7 US, Vintage Green/Outdoor Green-Desert Moss (921669-300)':'', ' Black/Anthracite-Black - 3.5 UK (DD9535-001)':'', ' Black/White-Black - 10 UK (DN3577-001)':'', "Boy's" : '', 'Quest 5 ':'Quest 5', 'M Air':'Air', 'MID NN':'Mid Nn', 'Strada)':'Strada'}
for old, new in replacement_dict.items():
    nike_running_df['Product Name'] = nike_running_df['Product Name'].str.replace(old, new, regex=False)

# drop any outstanding duplicates

nike_running_df = nike_running_df.drop_duplicates(subset=['Product Name', 'Actual Price (USD)'])

# remove shoe without a formal name

nike_running_df = nike_running_df.iloc[1:,:]

# print(nike_running_df)

def actual_price_bar_graph(df):
    plt.figure(figsize=(15, 8))
    sns.set_style('whitegrid')
    plot = sns.barplot(data=df, x='Product Name', y='Actual Price (USD)', color='#ff6f52')
    plt.subplots_adjust(top=0.85, bottom=0.18, left=0.083, right=0.871)
    plt.margins(x=0, y=0)
    labels = [textwrap.fill(label, 15) for label in df['Product Name']]
    plot.set_xticklabels(labels, rotation=45, horizontalalignment='center', fontsize=8.5, fontfamily='Calibri', fontweight='bold')
    plt.yticks([0,20,40,60,80,100,120,140],fontsize=14, fontfamily='Calibri',fontweight='bold')
    plt.ylim(0,140)
    for bar in plot.patches:
        bar.set_width(1)
    plt.title("Actual Price for Top-Rated Men's Nike Running Shoes", fontsize=26, fontweight='bold', pad=25)
    plt.xlabel('Running Shoe Name', fontsize=16, fontweight='bold', labelpad=15)
    plt.ylabel('Actual Price (USD)', fontsize=16, fontweight='bold', labelpad=10)
    plt.show()

actual_price_bar_graph(nike_running_df)

# use numpy to run pull some stats from the final prices

prices = nike_running_df['Actual Price (USD)']
mean_price = round(np.mean(prices), 2)
median_price = round(np.median(prices), 2)
std_dev_price = round(np.std(prices), 2)
min_price = round(np.min(prices), 2)
max_price = round(np.max(prices), 2)

print("Actual Price for Top-Rated Men's Nike Running Shoes --> Stats:")
print(f'Mean Price: ${mean_price}')
print(f'Median Price: ${median_price}')
print(f'Standard Deviation: ${std_dev_price}')
print(f'Minimum Price: ${min_price}')
print(f'Maximum Price: ${max_price}')





























