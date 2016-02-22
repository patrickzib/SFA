Scalable Time Series Data Analytics

Working with time series is difficult due to the high dimensionality of the data, erroneous or extraneous data, 
and large datasets. At the core of time series data analytics there are (a) a time series representation and (b) 
a similarity measure to compare two time series. There are many desirable properties of similarity measures1. 
Common similarity measures in the context of time series are Dynamic Time Warping (DTW) or the Euclidean Distance (ED). 
However, these are decades old and do not meet today’s requirements. We believe that the over-dependance of research on 
the UCR time series classification benchmark has led to two pitfalls, namely: (a) they focus mostly on accuracy2 and (b) 
they assume pre-processed datasets3. We identified three (additional) desirable properties: (a) alignment-free structural 
similarity, (b) noise-robustness, and (c) scalability.
We introduce a symbolic time series representation and two time series similarity measures for alignment-free, noise-robust
and scalable time series data analytics. 

# SFA: Symbolic Fourier Approximation

The symbolic time series representation Symbolic Fourier Approximation (SFA) represents each real-valued time series by a 
string. SFA is composed of approximation using the Fourier transform and quantization using a technique called Multiple 
Coefficient Binning (MCB). Among its properties, the most notable are: (a) noise removal due to low-pass filtering and 
quantization, (b) the string representation due to quantization, and (c) the frequency domain nature of the Fourier transform. 
The frequency domain nature makes SFA unique among the symbolic time series representations. Dynamically adding or removing 
Fourier coefficients to adapt the degree of approximation is at the core of the implemented algorithms.


![SFA](images/sfa_representation.png)

SFA was published in 

"Schäfer, P., Högqvist, M.: SFA: a symbolic fourier approximation and index for similarity search in high dimensional datasets. In: EDBT, ACM (2012)"
http://dl.acm.org/citation.cfm?doid=2247596.2247656


# BOSS: Bag-of-SFA-Symbols

Our Bag-Of-SFA-Symbols (BOSS) model combines the noise tolerance of the time series representation Symbolic Fourier 
Approximation (SFA) with the structure-based representation of the bag-of-words model which makes it inherently alignment-free. 
Apart from invariance to noise, the BOSS model provides invariances (robustness) to phase shifts, offsets, amplitudes and 
occlusions by discarding the original ordering of the SFA words and normalization. This leads to the highest classification 
and clustering accuracy in time series literature to date.

![SFA](images/classifiers.png)


BOSS was published in 

"Schäfer, P.: The boss is concerned with time series classification in the presence of noise. DMKD 29(6) (2015) 1505–1530"
http://link.springer.com/article/10.1007%2Fs10618-014-0377-7


# BOSS VS: Bag-of-SFA-Symbols in Vector Space

Our BOSS in Vector Space (BOSS VS) model builds upon the BOSS model for alignment-free and noise-robust time series data 
analytics combined with the vector space model (term frequency-inverse document frequency model). It significantly reduces 
the computational complexity of the BOSS model to allow for the classifi- cation of massive time series datasets. Its moderate 
train complexity, which is lower than the test complexity of 1-NN DTW, allows for frequent model updates such as mining streaming 
data (aka real-time predictive analytics). The 1-NN BOSS VS is not the most accurate classifier. However, its high speed combined 
with its good accuracy makes it unique and relevant for many practical use cases.

BOSS VS was published in 

"The Bag-of-SFA-Symbols in Vector Space classifier as published in Schäfer, P.: Scalable time series classification. DMKD (Preprint)"
http://link.springer.com/article/10.1007%2Fs10618-015-0441-y


# Acknowledgements

This work is supported by the [ZIB (Zuse Institute Berlin)](http://www.zib.de/en/home.html).

The UCR Time Series Classification Archive can be downloaded from:
http://www.cs.ucr.edu/~eamonn/time_series_data/
