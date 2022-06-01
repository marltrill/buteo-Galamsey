# Detection of Galamsey (illegal small-scale gold mining) in Ghana using a deep learning method based on Sentinel satellite imagery

Thesis Repository for Master of Science in Technology: Geoinformatics-  Aalborg University 

## About
This study was designed and built by Marcia Luz Trillo, with the help of NIRAS (1), in partial fulfillment of the degree of Master of Science in Technology at Aalborg University (2). This research was carried out using mainly Buteo Toolbox (3), QGIS (4) and GDAL commands (5). 
This thesis incorporates the acquired knowledge and practices gathered throughout the masters degree in an attempt to identify illegal small scale mining hotspots in southern Ghana.

(1) https://www.niras.com/
(2) https://www.en.aau.dk/
(3) https://github.com/casperfibaek/buteo
(4) https://www.qgis.org/en/site/
(5) https://gdal.org/

## Thesis Abstract
Ghana is considered one of the largest gold-producing countries in the world and is ranked first in the African continent. This nation produces vast quantities of gold through Galamsey (illegal small-scale gold mining), which is a very popular but unregulated technique for mineral extraction in southern Ghana and is the main source of income for many Ghanaians. Over the past decade, Galamsey has grown tremendously, causing a concerning and noticeable degradation of the environment and posing a real threat to people’s lives. Attempts have been made to detect and map these illicit mining operations in the past, but a high-quality map identifying the distribution patterns of Galamsey has not yet been conducted. In this study, a highly detailed Galamsey identification map of the entire country of Ghana was produced using an image recognition deep learning method. More specifically, the regression ML approach to detect Galamsey features followed a Convolutional Neural Network (CNN) algorithm with an Inception- ResNet -like architecture. Predictions were computed using the current best available resolution for open-source satellite images (10m), and Sentinel-1 and -2 products were processed to train the ML model for a three months period (November 1- 2021 to February 1- 2022). Even though further studies should include more algorithm testing, this pixel-based method delivered good results, achieving a binary accuracy of nearly 90%. Model predictions have shown that illegal mining is concentrated in four main regions in Ghana, being Western Ghana the hotspot for unlicensed artisanal miners. Galamsey spatial distribution is characterized by clusters along the ramification of main water streams of the country, degrading forest reserves that are protected at a national level. The ML method presented in this study serves as a valuable tool for identifying unauthorized gold mining activities and is therefore of considerable significance for government decision-makers and stakeholders that are involved in law-making practices against Galamsey.