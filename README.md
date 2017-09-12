# Time Series Data Analytics

Working with time series is difficult due to the high dimensionality of the data, erroneous or extraneous data, 
and large datasets. At the core of time series data analytics there are (a) a time series representation and (b) 
a similarity measure to compare two time series. There are many desirable properties of similarity measures. 
Common similarity measures in the context of time series are Dynamic Time Warping (DTW) or the Euclidean Distance (ED). 
However, these are decades old and do not meet today’s requirements. The over-dependence of research on
the UCR time series classification benchmark has led to two pitfalls, namely: (a) they focus mostly on accuracy and (b) 
they assume pre-processed datasets. There are additional desirable properties: (a) alignment-free structural 
similarity, (b) noise-robustness, and (c) scalability.

This repository contains a symbolic time series representation (**SFA**) and three time series models (**WEASEL**, **BOSS** and **BOSSVS**) for alignment-free, noise-robust and scalable time series data analytics. 

The implemented algorithms are in the context of:

1. **Dimensionality Reduction**: SFA performs significantly better than many other dimensionality reduction techniques including those techniques based on mean values like SAX, PLA, PAA, or APCA. This is due the fact, that SFA builds upon DFT, which is significantly more accurate than the other dimensionality reduction techniques [[1]](http://dl.acm.org/citation.cfm?doid=2247596.2247656).

2. **Classification and Accuracy**: WEASEL and the BOSS ensemble classifier offer state of art classification accuracy [[2]](http://arxiv.org/abs/1602.01711), [[3]](http://link.springer.com/article/10.1007%2Fs10618-014-0377-7), [[4]](https://arxiv.org/abs/1701.07681).

3. **Classification and Scalability**: WEASEL follows the bag-of-patterns approach which achieves highly competitive classification accuracies and is very fast, making it applicable in domains with high runtime and quality constraints. The novelty of WEASEL is its carefully engineered feature space using statistical feature selection, word co-occurrences, and a supervised symbolic representation for generating discriminative words. Thereby, WEASEL assigns high weights to characteristic, variable-length substructures of a TS. In our evaluation, WEASEL is consistently among the best and fastest methods, and competitors are either at the same level of quality but much slower or faster but much worse in accuracy. [[4]](https://arxiv.org/abs/1701.07681).
The BOSS VS classifier is one to four orders of magnitude faster than state of the art and significantly more accurate than the 1-NN DTW classifier, which serves as the benchmark to compare to. I.e., one can solve a classification problem with 1-NN DTW CV that runs on a cluster of 4000 cores for one day, with the BOSS VS classifier using commodity hardware and a 4 core cpu within one to two days resulting in a similar or better classification accuracy [[5]](http://link.springer.com/article/10.1007%2Fs10618-015-0441-y). 

![SFA](images/classifiers2.png)

Figure (second from left) shows the BOSS model as a histogram over SFA words. It first extracts subsequences (patterns) from a time series. Next, it applies low-pass filtering and quantization to the subsequences using SFA which reduces noise and allows for string matching algorithms to be applied. Two time series are then compared based on the differences in the histogram of SFA words.
 
Figure (second from right) illustrates the BOSS VS model. The BOSS VS model extends the BOSS model by a compact representation of classes instead of time series by using the term frequency - inverse document frequency (tf-idf) for each class. It significantly reduces the computational complexity and highlights characteristic SFA words by the use of the tf-idf weight matrix which provides an additional noise reducing effect.

Figure (right) illustrates the WEASEL model. WEASEL conceptually builds on the bag-of-patterns model. It derives discriminative features based on dataset labels. WEASEL extracts windows at multiple lengths and also considers the order of windows (using word co-occurrences as features) instead of considering each fixed-length window as independent feature (as in BOSS or BOSS VS). It then builds a single model from the concatenation of feature vectors. It finally applies an aggressive statistical feature selection to remove irrelevant features from each class. This resulting feature set is highly discriminative, which allows us to use fast logistic regression.

# Accuracy and Scalability

![TEST](images/walltime_predict_new3.png)

The figure shows for the state-of-the-art classifiers the total runtime on the x-axis in log scale vs the average rank on the y-axis for prediction. Runtimes include all preprocessing steps like feature extraction or selection. 

There are fast time series classifiers (BOSS VS, TSBF, LS, DTW CV) that require a few ms per prediction, but have a low average rank; and there are accurate methods (ST; BOSS; EE; COTE) that require hundredths of ms to seconds per prediction. The two ensemble methods in our comparison, EE PROP and COTE, show the highest prediction times. 

There is always a trade-off between accuracy and prediction times.  However, WEASEL is consistently among the best and fastest predicting methods, and competitors are (a) either at the same level of quality (COTE) but much slower or (b) faster but much worse in accuracy (LS, DTW CV, TSBF, or BOSS VS).

# SFA: Symbolic Fourier Approximation

The symbolic time series representation Symbolic Fourier Approximation (SFA) represents each real-valued time series by a 
string. SFA is composed of approximation using the Fourier transform and quantization using a technique called Multiple 
Coefficient Binning (MCB). Among its properties, the most notable are: (a) noise removal due to low-pass filtering and 
quantization, (b) the string representation due to quantization, and (c) the frequency domain nature of the Fourier transform. 
The frequency domain nature makes SFA unique among the symbolic time series representations. Dynamically adding or removing 
Fourier coefficients to adapt the degree of approximation is at the core of the implemented algorithms.


![SFA](images/sfa_representation.png)

The figure illustrates the SFA transformation. The time series is first Fourier transformed, low-pass filtered, and then quantized to its SFA word CBBCCDCBBCBCBEBED. Higher frequency components of a signal represent rapid changes, which are often associated with noise or dropouts. By keeping the first Fourier values, the signal is smoothened, equal to a low-pass filter. Quantization builds an envelope around the Fourier transform of the time series. Since symbolic representations are essentially a character string, they can be used with string algorithms and data structures such prefix tries, bag-of-words, Markov models, or string-matching.


**Usage:**

First, train the SFA quantization using a set of samples.

```java
int wordLength = 4;	// represents the length of the resulting SFA words. typically, between 4 and 16.
int symbols = 4; 	// symbols of the discretization alphabet. 4 is the default value

// Load datasets
TimeSeries[] train = TimeSeriesLoader.loadDatset(new File("./datasets/CBF/CBF_TEST"));

// Train the SFA representation
short[][] wordsTrain = sfa.fitTransform(train, wordLength, symbols);
```

Next, transform a time series using the trained quantization bins.

```java
// Transform a times series
TimeSeries ts = ...;

// DFT approximation of the time series
double[] dftTs = sfa.transformation.transform(ts, ts.getLength(), wordLength);

// SFA quantization to an SFA word
short[] wordTs = sfa.quantization(dftTs);
```

Similarity search using the SFA distance.

```java
boolean normMean = true or false; // set to true, if mean should be set to 0 for a window

double distance = sfaDistance.getDistance(wordsTrain[t], wordTs, dftTs, normMean, minDistance);
  
// check the real distance, if the lower bounding distance is smaller than the best-so-far distance
if (distance < minDistance) {
  double realDistance = getEuclideanDistance(train[t], ts, minDistance);
  if (realDistance < minDistance) {
    minDistance = realDistance;
    best = t;
  }
}
```

Similarity search using the index SFATrie.

```java
int l = 16; 	// SFA word length ( & dimensionality of the index)
int windowLength = 256; 	// length of the subsequences to be indexed
int leafThreshold = 100; 	// number of subsequences in each leaf node
int k = 10; // k-NN search

// Load datasets
TimeSeries timeSeries = ...;
TimeSeries query = ...;

// create the SFA trie & and index the time series
SFATrie index = new SFATrie(l, leafThreshold);
index.buildIndex(timeSeries, windowLength);

// perform a k-NN search
SortedListMap<Double, Integer> result = index.searchNearestNeighbor(query, k);

// output result
List<Integer> offsets = result.values();
List<Double> distances = result.keys();

for (int j = 0; j < result.size(); j++) {
	System.out.println("\tResult:\t"+distances.get(j) + "\t" +offsets.get(j));
}

```


**References**

"Schäfer, P., Högqvist, M.: SFA: a symbolic fourier approximation and index for similarity search in high dimensional datasets. In: EDBT, ACM (2012)"
[[LINK]](http://dl.acm.org/citation.cfm?doid=2247596.2247656)


# BOSS: Bag-of-SFA-Symbols

The Bag-Of-SFA-Symbols (BOSS) model combines the noise tolerance of the time series representation Symbolic Fourier 
Approximation (SFA) with the structure-based representation of the bag-of-words model which makes it inherently alignment-free. 
Apart from invariance to noise, the BOSS model provides invariances (robustness) to phase shifts, offsets, amplitudes and 
occlusions by discarding the original ordering of the SFA words and normalization. This leads to the highest classification 
and clustering accuracy in time series literature to date.

**Usage:**

First, to train the BOSS model using a set of samples, we first have to obtain the SFA words:

```java
boolean normMean = true or false; // set to true, if mean should be set to 0 for a window
int maxF = 4;	// represents the length of the resulting SFA words. typically, in-between 4 and 16.
int maxS = 4; 	// symbols of the discretization alphabet. 4 is the default value
// subsequence (window) length used for extracting SFA words from time series. 
// typically, in-between 4 and time series length n.
int windowLength = ...; 

TimeSeries[] trainSamples = ...

BOSSVSModel model = new BOSSVSModel(maxF, maxS, windowLength, normMean);
int[][] words = model.createWords(trainSamples);
```

Next, we build a histogram of word frequencies (bag-of-patterns):

```java
BagOfPattern[] bag = model.createBagOfPattern(words, trainSamples, wordLength);
```

**References**

"Schäfer, P.: The BOSS is concerned with time series classification in the presence of noise. DMKD (2015)"
[[LINK]](http://link.springer.com/article/10.1007%2Fs10618-014-0377-7)


# BOSS VS: Bag-of-SFA-Symbols in Vector Space

The BOSS in Vector Space (BOSS VS) model builds upon the BOSS model for alignment-free and noise-robust time series data 
analytics combined with the vector space model (term frequency-inverse document frequency model). It significantly reduces 
the computational complexity of the BOSS model to allow for the classifi- cation of massive time series datasets. Its moderate 
train complexity, which is lower than the test complexity of 1-NN DTW, allows for frequent model updates such as mining streaming 
data (aka real-time predictive analytics). The BOSS VS is not the most accurate classifier. However, its high speed combined 
with its good accuracy makes it unique and relevant for many practical use cases.

**Usage:**

First, to train the BOSS VS model using a set of samples, we first have to obtain the SFA words:

```java
boolean normMean = true or false; // set to true, if mean should be set to 0 for a window
int maxF = 4;	// represents the length of the resulting SFA words. typically, in-between 4 and 16.
int maxS = 4; 	// symbols of the discretization alphabet. 4 is the default value
// subsequence (window) length used for extracting SFA words from time series. 
// typically, in-between 4 and time series length n.
int windowLength = ...; 

TimeSeries[] trainSamples = ...

BOSSVSModel model = new BOSSVSModel(maxF, maxS, windowLength, normMean);
int[][] words = model.createWords(trainSamples);
```

Next, we build a histogram of word frequencies (bag-of-patterns):

```java
BagOfPattern[] bag = model.createBagOfPattern(words, trainSamples, wordLength);
```

Finally, we build obtain the tf-idf model from the bag-of-patterns for each class label (uniqueLabels):

```java
ObjectObjectHashMap<String, IntFloatHashMap> idf = model.createTfIdf(bag, uniqueLabels);
```

**References**

"Schäfer, P.: Scalable Time Series Classification. DMKD (2016) and ECML/PKDD 2016
[[LINK]](http://link.springer.com/article/10.1007%2Fs10618-015-0441-y)


# WEASEL: Word ExtrAction for time SEries cLassification

The Word ExtrAction for time SEries cLassification (WEASEL) model builds upon the bag-of-pattern model. The novelty of WEASEL lies in its specific method for deriving features, resulting in a much smaller yet much more discriminative feature set. WEASEL is more accurate than the best current non-ensemble algorithms at orders-of-magnitude lower classification and training times. WEASEL derives discriminative features based on the dataset labels. WEASEL extracts windows at multiple lengths and also considers the order of windows (using word co-occurences as features) instead of considering each fixed-length window as independent feature. It then builds a single model from the concatenation of feature vectors. So, instead of training O(n) different models, and picking the best one, we weigh each feature based on its relevance to predict the class. Finally, WEASEL applies an aggressive statistical feature selection to remove irrelevant features from each class, without negatively impacting accuracy and heavily reducing runtime. 

**Usage:**

First, to train the WEASEL model using a set of samples, we first have to obtain the supervised SFA words:

```java
boolean normMean = true or false; // set to true, if mean should be set to 0 for a window
int wordLength = 4;	// represents the length of the resulting SFA words. typically, in-between 4 and 16.
int maxS = 4; 		// symbols of the discretization alphabet. 4 is the default value
// range of window lengths to use for extracting SFA words from time series. 
// typically, set to all window lengths in-between 4 and n.
int[] windowLengths = new int[]{...}; 
	

TimeSeries[] trainSamples = ...

WEASELModel model = new WEASELModel(wordLength, maxS, windowLengths, normMean, false);
int[][][] words = model.createWords(samples);
```

Next, we build a histogram of word co-occurrences (bi-grams) frequencies (bag-of-bigrams):

```java
BagOfBigrams[] bop = model.createBagOfPatterns(words, samples, wordLength);
```

Next, we apply the chi-squared test to remove irrelevant features from each class:

```java
model.filterChiSquared(bop, chi);
```

Finally, we train the logistic regression classifier (using default parameters of liblinear) to assign high weights to discriminative words of each class.

```java
final Problem problem = initLibLinearProblem(bop, model.dict, -1);
int correct = trainLibLinear(problem, SolverType.L2R_LR_DUAL, 1, 5000, 0.1, 10, new Random(1));
```


**References**

"Schäfer, P., Leser, U.: Fast and Accurate Time Series Classification with WEASEL."
CIKM 2017, (accepted), [[LINK]](https://arxiv.org/abs/1701.07681)


# Use Cases / Tests

There are 7 implemented use cases:


1. Classification accuracy of WEASEL, BOSS VS and BOSS ensemble on the UCR datasets: [UCRClassification.java](https://github.com/patrickzib/SFA/blob/master/src/sfa/test/UCRClassification.java)
2. SFA lower bounding distance to the Euclidean distance: 
[SFAMinDistance.java](https://github.com/patrickzib/SFA/blob/master/src/sfa/test/SFAMinDistance.java)
3. Generate SFA words from a set of samples: 
[SFAWords.java](https://github.com/patrickzib/SFA/blob/master/src/sfa/test/SFAWords.java)
4. SFA makes use of variable word lengths: 
[SFAWordsVariableLength.java](https://github.com/patrickzib/SFA/blob/master/src/sfa/test/SFAWordsVariableLength.java)
5. Extract sliding windows from a time series and transform each sliding window to its SFA word: 
[SFAWordsWindowing.java](https://github.com/patrickzib/SFA/blob/master/src/sfa/test/SFAWordsWindowing.java)
6. Time series indexing and similarity search: 
[SFATrieTest.java](https://github.com/patrickzib/SFA/blob/master/src/sfa/test/SFATrieTest.java)
7. Time series bulk loading and similarity search: 
[SFABulkLoad.java](https://github.com/patrickzib/SFA/blob/master/src/sfa/test/SFABulkLoad.java)


# References & Acknowledgements

This work is supported by the [ZIB (Zuse Institute Berlin)](http://www.zib.de/en/home.html) and [HU Berlin (Humboldt-Universität zu Berlin)](http://www.hu-berlin.de).

Read more about Scalable Time Series Data Analytics in the [Dissertation](http://edoc.hu-berlin.de/docviews/abstract.php?id=42117).

[The UCR Time Series Classification Archive](http://www.cs.ucr.edu/~eamonn/time_series_data/)

[The Great Time Series Classification Bake Off: An Experimental Evaluation of Recently Proposed Algorithms. Extended Version](http://arxiv.org/abs/1602.01711)

Many thanks to @ChristianSch for porting the project to gradle.

