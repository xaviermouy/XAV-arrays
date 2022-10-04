# Identification of fish sounds in the wild using a set of portable audio-video arrays

This repository contains the necessary information to assemble, operate, deploy, and process data from three eXperimental Audio-Video (XAV) arrays used for identifying fish sounds in the wild.

## Building and deploying the audio-video arrays
Building instructions and suggested deployment procedures can be found in the Supplementary Information of the paper. Deploymnet log sheets and and check lists can be found in the logsheets folder [here](https://github.com/xaviermouy/XAV-arrays/tree/main/logsheets/).

## Sample data
Two small sets of acoustic data from the large and mobile arrays are provided in this repository and used to show how the acoutsic localization works (see section below). All the acoustic data used in the paper can be found on teh OSF data repository of the paper.

* **Data from the large array**: [here](https://github.com/xaviermouy/XAV-arrays/tree/main/localization/large-array)
* **Data from the mobile array**: [here](https://github.com/xaviermouy/XAV-arrays/tree/main/localization/mobile-array)

## Acoustic localization
Acoustic localization in performed using either linearized inversion or fully non-linear inversion. We provide python code for both approaches. 

### Linearized inversion
The localization process using linearized inversion is described in this Jupyter Notebook using data from the large audio-video array.
The notebook can also be executed on Google Colab.

* **Jupyter Notebook**: here
* **GoogleColab Notebook**: here

### Fully non-linear inversion
The localization process using fully non-limear inversion is described in this Jupyter Notebook using data from the mobile audio-video array.
The notebook can also be executed on Google Colab.

* **Jupyter Notebook**: here
* **GoogleColab Notebook**: here

## Simulated annealing for optimizing the placement of the hydrophones

* **Jupyter Notebook**: here
* **GoogleColab Notebook**: here

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/bJMbtHWPlEg/0.jpg)](http://www.youtube.com/watch?v=bJMbtHWPlEg)

## Status
This page is still under construction...
 
