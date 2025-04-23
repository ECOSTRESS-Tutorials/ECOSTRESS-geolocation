# ECOSTRESS_Geolocation_Correction_tool
This code is designed to find and apply pixel shifts to ECOSTRESS LST images to match Sentinel-2 accuracy.

Please navigate to the [ECOSTRESS Tutorials Repository](https://github.com/ECOSTRESS-Tutorials) to familiarize yourself with ECOSTRESS products .


Overview 

ECOSTRESS relies on information from the International Space Station (ISS) for geolocation coordinates, which could be off by as much as 7km. Within ECOSTRESS geolocation processing, image matching is used to improve geolocation, although this is not always successful. ECOSTRESS scenes located near bodies of water have noticeable errors as observers can easily find discrepancies between a base map and the ECOSTRESS scene. This code aims to use ECOSTRESS’s water mask product to correctly place the ECOSTRESS file to a better position. In this repository there is a **jupyter notebook** with an single file example of how the code works. The **File Name** is the main code where a folder directory could be inserted and the code will run for every possible scene that has an LST, QC, and Water Mask file. 


Questions, suggestions, or remarks: aalamillo619@gmail.com, madeleine.a.pascolini-campbell@jpl.nasa.gov, caroline.r.baumann@gmail.com
