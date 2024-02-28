import pandas as pd

from termcolor import colored

import numpy as np

#Visualization libraries
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
# import plotly.express as px
# import plotly.graph_objs as go
# from plotly import tools
# from plotly.offline import iplot
#
# #Geospatial Analysis Libraries
# import geopandas as gpd
import math
import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
#
# #NLTK libraries
# import nltk
# import re
# import string
# from wordcloud import WordCloud,STOPWORDS
# from nltk.corpus import stopwords
# from textblob import TextBlob
#
#
# #Miscellaneous libraries
# from collections import defaultdict
# import cufflinks as cf
# cf.go_offline()
# cf.set_config_file(offline=False, world_readable=True)


def preliminar_analysis(delivery):
    print("****************************************************************")
    print("******************* Preliminar analysis ************************")
    print("****************************************************************")
    # Printing the information of dataset
    print("[INFO] The shape of the  data is (row, column):" + str(delivery.shape))
    print("########################################################")
    print("[INFO] Dataset info:")
    print(delivery.info())
    print("########################################################")
    print("[INFO] Head:\n{}".format(delivery.head()))
    print("########################################################")
    print("[INFO] Dataset describe:")
    print(delivery.describe())
    return


def demographic_analysis(delivery):
    print("****************************************************************")
    print("******************* Demographic analysis ***********************")
    print("****************************************************************")
    # Pivot table
    delivery_pivot1 = pd.pivot_table(delivery, index=["Gender", "Marital Status"],
                                     values=['Age', 'Family size'],
                                     aggfunc=[np.mean, len], margins=True)

    # Adding color gradient
    cm = sns.light_palette("green", as_cmap=True)
    delivery_pivot1.style.background_gradient(cmap=cm)
    print(delivery_pivot1)
    print("########################################################")

    # Pivot table
    delivery_pivot2 = pd.pivot_table(delivery, index=["Educational Qualifications", "Occupation"],
                                     values=['Age', 'Family size'],
                                     aggfunc=[np.mean, len, np.std])

    # Adding bar for numbers
    delivery_pivot2.style.bar()
    print(delivery_pivot2)
    print("########################################################")

    # Pivot table
    delivery_pivot3 = pd.pivot_table(delivery, index=["Occupation", "Monthly Income"],
                                     values=['Age', 'Family size'],
                                     aggfunc=[np.mean, len, np.std])

    # Adding style
    delivery_pivot3.style \
        .format('{:.2f}') \
        .bar(align='mid', color=['lightgreen']) \
        .set_properties(padding='5px', border='3px solid white', width='200px')
    print(delivery_pivot3)
    print("########################################################")

    # Setting up the frame
    plt.figure(figsize=(15, 7))
    plt.style.use('seaborn-white')

    # Gender Countplot
    plt.subplot(2, 3, 1)
    ax = sns.countplot(x="Gender", data=delivery,
                       facecolor=(0, 0, 0, 0),
                       linewidth=5,
                       edgecolor=sns.color_palette("dark", 3))
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
    ax.set_title('Gender count', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count_Gender', fontsize=12)
    plt.tight_layout()

    # Marital Status Countplot
    plt.subplot(2, 3, 2)
    ax = sns.countplot(x="Marital Status", data=delivery,
                       facecolor=(0, 0, 0, 0),
                       linewidth=5,
                       edgecolor=sns.color_palette("dark", 3))
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
    ax.set_title('Marital Status count', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count_Marital', fontsize=12)
    plt.tight_layout()

    # Occupation Countplot
    plt.subplot(2, 3, 3)
    ax = sns.countplot(x="Occupation", data=delivery,
                       facecolor=(0, 0, 0, 0),
                       linewidth=5,
                       edgecolor=sns.color_palette("dark", 3))
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
    ax.set_title('Occupation count', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count_Occupation', fontsize=12)
    plt.tight_layout()

    # Education Countplot
    plt.subplot(2, 3, 4)
    ax = sns.countplot(x="Educational Qualifications", data=delivery,
                       facecolor=(0, 0, 0, 0),
                       linewidth=5,
                       edgecolor=sns.color_palette("dark", 3))
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
    ax.set_title('Educational Qualifications count', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count_Ed', fontsize=12)
    plt.tight_layout()

    # Income Countplot
    plt.subplot(2, 3, 5)
    ax = sns.countplot(x="Monthly Income", data=delivery,
                       facecolor=(0, 0, 0, 0),
                       linewidth=5,
                       edgecolor=sns.color_palette("dark", 3))
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=20)
    ax.set_title('Monthly Income count', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count_Income', fontsize=12)
    plt.tight_layout()

    # Family Size Countplot
    plt.subplot(2, 3, 6)
    ax = sns.countplot(x="Family size", data=delivery,
                       facecolor=(0, 0, 0, 0),
                       linewidth=5,
                       edgecolor=sns.color_palette("dark", 3))
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=20)
    ax.set_title('Family size', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count_Size', fontsize=12)
    plt.tight_layout()

    plt.savefig("outputs/demographic.png")
    plt.close()
    return



def delivery_time_analysis(delivery):
    print("****************************************************************")
    print("****************** Delivery time analysis **********************")
    print("****************************************************************")
    # Pivot table
    delivery_pivot4 = pd.pivot_table(delivery, index=["Order Time", "Maximum wait time"],
                                     values=['Age', 'Family size'], columns=['Influence of time'],
                                     aggfunc={'Influence of time': len},
                                     fill_value=0)

    # Adding color gradient
    cm = sns.light_palette("blue", as_cmap=True)
    delivery_pivot4.style.background_gradient(cmap=cm)
    print(delivery_pivot4)
    print("########################################################")

    # Pivot table
    delivery_pivot5 = pd.pivot_table(delivery, index=["Medium (P1)", "Perference(P1)"],
                                     columns=['Influence of time'],
                                     aggfunc={'Influence of time': len},
                                     fill_value=0)
    # Adding color gradient
    delivery_pivot5.style \
        .format('{:.2f}') \
        .bar(align='mid', color=['lightblue']) \
        .set_properties(padding='5px', border='3px solid white', width='200px')
    print(delivery_pivot5)
    print("########################################################")
    return


def univariate_expl_analysis(delivery):
    # Setting up the frame
    fig, axes = plt.subplots(nrows=2, ncols=2, dpi=120, figsize=(8, 6))

    # Distribution of age with displot
    plot00 = sns.histplot(delivery['Age'], ax=axes[0][0], color='green', kde=True)
    axes[0][0].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    axes[0][0].set_title('Distribution of Age', fontdict={'fontsize': 8})
    axes[0][0].set_xlabel('Age', fontdict={'fontsize': 7})
    axes[0][0].set_ylabel('Count/Dist.', fontdict={'fontsize': 7})
    plt.tight_layout()

    # Distribution of Familysize with displot
    plot01 = sns.histplot(delivery['Family size'], ax=axes[0][1], color='green', kde=True)
    axes[0][1].set_title('Distribution of Family Size', fontdict={'fontsize': 8})
    axes[0][1].set_xlabel('Family Size', fontdict={'fontsize': 7})
    axes[0][1].set_ylabel('Count/Dist.', fontdict={'fontsize': 7})
    axes[0][1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    plt.tight_layout()

    # Age-Boxplot
    plot10 = sns.boxplot(data=delivery['Age'], ax=axes[1][0])
    axes[1][0].set_title('Age Distribution', fontdict={'fontsize': 8})
    axes[1][0].set_xlabel('Distribution', fontdict={'fontsize': 7})
    axes[1][0].set_ylabel(r'Age', fontdict={'fontsize': 7})
    plt.tight_layout()

    # FamilySize-Boxplot
    plot11 = sns.boxplot(data=delivery['Family size'], ax=axes[1][1])
    axes[1][1].set_title(r'Numerical Summary (Family Size)', fontdict={'fontsize': 8})
    axes[1][1].set_ylabel(r'Size of Family', fontdict={'fontsize': 7})
    axes[1][1].set_xlabel('Family Size', fontdict={'fontsize': 7})
    plt.tight_layout()

    plt.savefig("outputs/univ_age_famsize.png")
    plt.close()
    return


def consumer_preferences(delivery):
    # Setting up the frame
    plt.figure(figsize=(15, 7))
    plt.style.use('seaborn-white')

    # Meal Countplot
    plt.subplot(1, 3, 1)
    ax = sns.countplot(x="Meal(P1)", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
    ax.set_title('Meal count', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count_Meal', fontsize=12)
    plt.tight_layout()

    # Medium Countplot
    plt.subplot(1, 3, 2)
    ax = sns.countplot(x="Medium (P1)", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11)
    ax.set_title('Medium Status count', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count_Medium', fontsize=12)
    plt.tight_layout()

    # Preference Countplot
    plt.subplot(1, 3, 3)
    ax = sns.countplot(x="Perference(P1)", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=20)
    ax.set_title('Preference count', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count_Preference', fontsize=12)
    plt.tight_layout()

    plt.savefig("outputs/consum_pref.png")
    plt.close()
    return


def purchase_demand(delivery):
    # Setting up the frame
    plt.figure(figsize=(15, 7))
    plt.style.use('ggplot')

    # Ease and convenient Countplot
    plt.subplot(2, 4, 1)
    ax = sns.countplot(x="Ease and convenient", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('Ease and convenient', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()

    # Time Countplot
    plt.subplot(2, 4, 2)
    ax = sns.countplot(x="Time saving", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('Time saving', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()

    # Restaurant Countplot
    plt.subplot(2, 4, 3)
    ax = sns.countplot(x="More restaurant choices", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('More restaurant choices', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()

    # Payment Countplot
    plt.subplot(2, 4, 4)
    ax = sns.countplot(x="Easy Payment option", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('Easy Payment option', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()

    # Offers Countplot
    plt.subplot(2, 4, 5)
    ax = sns.countplot(x="More Offers and Discount", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('More Offers and Discount', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()

    # Preference Countplot
    plt.subplot(2, 4, 6)
    ax = sns.countplot(x="Good Food quality", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('Good Food quality', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count_Preference', fontsize=12)
    plt.tight_layout()

    # Tracking Countplot
    plt.subplot(2, 4, 7)
    ax = sns.countplot(x="Good Tracking system", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('Good Tracking system', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()

    plt.savefig("outputs/purchase_demand.png")
    plt.close()
    return


def concerns(delivery):
    # Setting up the frame
    plt.figure(figsize=(15, 7))
    plt.style.use('seaborn-dark')

    # Self cooking Countplot
    plt.subplot(2, 4, 1)
    ax = sns.countplot(x="Self Cooking", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('Self Cooking', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()

    # Health Countplot
    plt.subplot(2, 4, 2)
    ax = sns.countplot(x="Health Concern", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('Health Concern', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()

    # Late delivery Countplot
    plt.subplot(2, 4, 3)
    ax = sns.countplot(x="Late Delivery", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('Late Delivery', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()

    # Hygiene Countplot
    plt.subplot(2, 4, 4)
    ax = sns.countplot(x="Poor Hygiene", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('Poor Hygiene', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()

    # Past Countplot
    plt.subplot(2, 4, 5)
    ax = sns.countplot(x="Bad past experience", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('Bad past experience', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()

    # Unavailability Countplot
    plt.subplot(2, 4, 6)
    ax = sns.countplot(x="Unavailability", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('Unavailability', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count_Preference', fontsize=12)
    plt.tight_layout()

    # Unaffordable Countplot
    plt.subplot(2, 4, 7)
    ax = sns.countplot(x="Unaffordable", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('Unaffordable', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()

    plt.savefig("outputs/concerns.png")
    plt.close()
    return


def cancellations(delivery):
    # Setting up the frame
    plt.figure(figsize=(15, 7))
    plt.style.use('fivethirtyeight')

    # Long delivery time Countplot
    plt.subplot(2, 3, 1)
    ax = sns.countplot(x="Long delivery time", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('Long delivery time', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()

    # Delay of delivery person getting assigned Countplot
    plt.subplot(2, 3, 2)
    ax = sns.countplot(x="Delay of delivery person getting assigned", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('Delay of delivery person getting assigned', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()

    # Delay of delivery person picking up food Countplot
    plt.subplot(2, 3, 3)
    ax = sns.countplot(x="Delay of delivery person picking up food", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('Delay of delivery person picking up food', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()

    # Wrong order delivered Countplot
    plt.subplot(2, 3, 4)
    ax = sns.countplot(x="Wrong order delivered", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('Wrong order delivered', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()

    # Missing item Countplot
    plt.subplot(2, 3, 5)
    ax = sns.countplot(x="Missing item", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('Missing item', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()

    # Order placed by mistake Countplot
    plt.subplot(2, 3, 6)
    ax = sns.countplot(x="Order placed by mistake", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('Order placed by mistake', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count_Preference', fontsize=12)
    plt.tight_layout()

    plt.savefig("outputs/cancellations.png")
    plt.close()
    return


def time_factor(delivery):
    # Setting up the frame
    plt.figure(figsize=(15, 7))
    plt.style.use('bmh')

    # Influence of time Countplot
    plt.subplot(2, 4, 1)
    ax = sns.countplot(x="Influence of time", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('Influence of time', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()

    # Order Time Countplot
    plt.subplot(2, 4, 2)
    ax = sns.countplot(x="Order Time", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('Order Time', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()

    # Maximum wait time Countplot
    plt.subplot(2, 4, 3)
    ax = sns.countplot(x="Maximum wait time", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('Maximum wait time', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()

    # Hygiene Countplot
    plt.subplot(2, 4, 4)
    ax = sns.countplot(x="Residence in busy location", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('Residence in busy location', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()

    # Accuracy Countplot
    plt.subplot(2, 4, 5)
    ax = sns.countplot(x="Google Maps Accuracy", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('Google Maps Accuracy', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()

    # Good Road Condition Countplot
    plt.subplot(2, 4, 6)
    ax = sns.countplot(x="Good Road Condition", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('Good Road Condition', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()

    # Low quantity low time Countplot
    plt.subplot(2, 4, 7)
    ax = sns.countplot(x="Low quantity low time", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('Low quantity low time', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()

    # Delivery person ability Countplot
    plt.subplot(2, 4, 8)
    ax = sns.countplot(x="Delivery person ability", data=delivery)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax.set_title('Delivery person ability', fontsize=15)
    ax.set_xlabel('Types', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    plt.tight_layout()

    plt.savefig("outputs/time_factor.png")
    plt.close()
    return


def bivariate_time_others(delivery):
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    fig.suptitle('Bivariate Analysis-1')

    sns.boxplot(ax=axes[0], data=delivery, x='Influence of time', y='Age')
    sns.boxplot(ax=axes[1], data=delivery, x='Influence of time', y='Family size')
    sns.countplot(ax=axes[2], data=delivery, x="Occupation", hue="Influence of time")

    plt.savefig("outputs/biv_time_others.png")
    plt.close()
    return

def bivariate_2(delivery):
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    fig.suptitle('Bivariate Analysis-2')

    sns.countplot(ax=axes[0], data=delivery, x="Marital Status", hue="Maximum wait time")
    ax = sns.countplot(ax=axes[1], data=delivery, x="Monthly Income", hue="Influence of rating")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)
    ax = sns.countplot(ax=axes[2], data=delivery, x="Good Road Condition", hue="Long delivery time")
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=11, rotation=40)

    plt.savefig("outputs/biv_2.png")
    plt.close()
    return


def geospatial_analysis(delivery):
    Age_band = delivery[(delivery.Age.isin(range(18, 40)))]
    # Creating a map
    m_2 = folium.Map(location=[12.9716, 77.5946], tiles='cartodbpositron', zoom_start=13)

    # Adding points to the map
    for idx, row in Age_band.iterrows():
        Marker([row['latitude'], row['longitude']]).add_to(m_2)

    # Displaying the map
    m_2.save("outputs/order_map.html")

    # Creating the map
    m_3 = folium.Map(location=[12.9716, 77.5946], tiles='cartodbpositron', zoom_start=13)

    # Adding points to the map
    mc = MarkerCluster()
    for idx, row in Age_band.iterrows():
        if not math.isnan(row['longitude']) and not math.isnan(row['latitude']):
            mc.add_child(Marker([row['latitude'], row['longitude']]))
    m_3.add_child(mc)

    # Displaying the map
    m_3.save("outputs/order_clustered_map.html")


def main():
    delivery = pd.read_csv("data/onlinedeliverydata.csv")
    print(colored("read_csv: ", "magenta", attrs=["bold"]) + colored("DONE", "green", attrs=["bold"]),flush=True)

    preliminar_analysis(delivery)
    print(colored("Preliminar analysis: ", "magenta", attrs=["bold"]) + colored("DONE", "green", attrs=["bold"]),flush=True)

    demographic_analysis(delivery)
    print(colored("Demographic analysis: ", "magenta", attrs=["bold"]) + colored("DONE", "green", attrs=["bold"]),flush=True)

    delivery_time_analysis(delivery)
    print(colored("Delivery time analysis: ", "magenta", attrs=["bold"]) + colored("DONE", "green", attrs=["bold"]),flush=True)

    univariate_expl_analysis(delivery)
    print(colored("Univariate exploratory analysis: ", "magenta", attrs=["bold"]) + colored("DONE", "green", attrs=["bold"]),flush=True)

    consumer_preferences(delivery)
    print(colored("Consumer preference analysis: ", "magenta", attrs=["bold"]) + colored("DONE", "green", attrs=["bold"]),flush=True)

    purchase_demand(delivery)
    print(colored("Purchase demand analysis: ", "magenta", attrs=["bold"]) + colored("DONE", "green", attrs=["bold"]),flush=True)

    concerns(delivery)
    print(colored("Concerns analysis: ", "magenta", attrs=["bold"]) + colored("DONE", "green", attrs=["bold"]),flush=True)

    cancellations(delivery)
    print(colored("Cancellations analysis: ", "magenta", attrs=["bold"]) + colored("DONE", "green", attrs=["bold"]),flush=True)

    time_factor(delivery)
    print(colored("Time factor analysis: ", "magenta", attrs=["bold"]) + colored("DONE", "green", attrs=["bold"]),flush=True)

    bivariate_time_others(delivery)
    print(colored("Bivariate time-others analysis: ", "magenta", attrs=["bold"]) + colored("DONE", "green", attrs=["bold"]),flush=True)

    bivariate_2(delivery)
    print(colored("Bivariate 2 analysis: ", "magenta", attrs=["bold"]) + colored("DONE", "green", attrs=["bold"]),flush=True)

    geospatial_analysis(delivery)
    print(colored("Geospatial analysis: ", "magenta", attrs=["bold"]) + colored("DONE", "green", attrs=["bold"]),flush=True)


    return



if __name__ == "__main__":
    # This code is 99% based on the one at:
    # https://www.kaggle.com/code/benroshan/food-delivery-eda-starter
    main()