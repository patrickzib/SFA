// Copyright (c) 2017 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.transformation;

import com.carrotsearch.hppc.*;
import com.carrotsearch.hppc.cursors.LongFloatCursor;
import com.carrotsearch.hppc.cursors.LongIntCursor;
import sfa.classification.Classifier.Words;
import sfa.classification.ParallelFor;
import sfa.classification.WEASELClassifier;
import sfa.timeseries.TimeSeries;

import java.lang.reflect.Array;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The WEASEL-Model as published in
 * <p>
 * Schäfer, P., Leser, U.: Fast and Accurate Time Series
 * Classification with WEASEL. CIKM 2017
 */
public class WEASEL {

  public int alphabetSize;
  public int maxF;

  public int[] windowLengths;
  public boolean normMean;
  public boolean lowerBounding;
  public SFA[] signature;
  public Dictionary dict;

  public final static int BLOCKS;

  static {
    Runtime runtime = Runtime.getRuntime();
    if (runtime.availableProcessors() <= 4) {
      BLOCKS = 8;
    } else {
      BLOCKS = runtime.availableProcessors();
    }

    //    BLOCKS = 1; // for testing purposes
  }
  public WEASEL(){}

  /**
   * Create a WEASEL model.
   *
   * @param maxF          Length of the SFA words
   * @param maxS          alphabet size
   * @param windowLengths the set of window lengths to use for extracting SFA words from
   *                      time series.
   * @param normMean      set to true, if mean should be set to 0 for a window
   * @param lowerBounding set to true, if the Fourier transform should be normed (typically
   *                      used to lower bound / mimic Euclidean distance).
   */
  public WEASEL(
      int maxF, int maxS,
      int[] windowLengths, boolean normMean, boolean lowerBounding) {
    this.maxF = maxF;
    this.alphabetSize = maxS;
    this.windowLengths = windowLengths;
    this.normMean = normMean;
    this.lowerBounding = lowerBounding;
    this.dict = new Dictionary();
    this.signature = new SFA[windowLengths.length];
  }

  /**
   * The WEASEL-model: a histogram of SFA word and bi-gram frequencies
   */
  public static class BagOfBigrams {
    public LongIntHashMap bob;
    public Double label;

    public BagOfBigrams(int size, Double label) {
      this.bob = new LongIntHashMap(size);
      this.label = label;
    }
  }

  /**
   * Create SFA words and bigrams for all samples
   *
   * @param samples
   * @return
   */
  public int[][][] createWords(final TimeSeries[] samples) {
    // create bag of words for each window queryLength
    final int[][][] words = new int[this.windowLengths.length][samples.length][];
    ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        for (int w = 0; w < WEASEL.this.windowLengths.length; w++) {
          if (w % BLOCKS == id) {
            words[w] = createWords(samples, w);
          }
        }
      }
    });
    return words;
  }

  /**
   * Create SFA words and bigrams for all samples
   *
   * @param samples
   * @return
   */
  public int[][] createWords(final TimeSeries[] samples, final int index) {

    // SFA quantization
    if (this.signature[index] == null) {
      this.signature[index] = new SFASupervised();
      this.signature[index].fitWindowing(
          samples, this.windowLengths[index], this.maxF, this.alphabetSize, this.normMean, this.lowerBounding);
    }

    // create words
    final int[][] words = new int[samples.length][];
    for (int i = 0; i < samples.length; i++) {
      if (samples[i].getLength() >= this.windowLengths[index]) {
        words[i] = this.signature[index].transformWindowingInt(samples[i], this.maxF);
      } else {
        words[i] = new int[]{};
      }
    }

    return words;
  }

  /**
   * Create words and bi-grams for all window lengths
   */
  public BagOfBigrams[] createBagOfPatterns(
      final int[][] wordsForWindowLength,
      final TimeSeries[] samples,
      final int w,    // index of used windowSize
      final int wordLength) {
    BagOfBigrams[] bagOfPatterns = new BagOfBigrams[samples.length];

    final byte usedBits = (byte) Words.binlog(this.alphabetSize);
    final long mask = (1L << (usedBits * wordLength)) - 1L;
    int highestBit = Words.binlog(Integer.highestOneBit(WEASELClassifier.MAX_WINDOW_LENGTH))+1;

    // iterate all samples
    // and create a bag of pattern
    for (int j = 0; j < samples.length; j++) {
      bagOfPatterns[j] = new BagOfBigrams(wordsForWindowLength[j].length * 2, samples[j].getLabel());

      // create subsequences
      for (int offset = 0; offset < wordsForWindowLength[j].length; offset++) {
        long word = (wordsForWindowLength[j][offset] & mask) << highestBit | (long) w;
        bagOfPatterns[j].bob.putOrAdd(word, 1, 1);

        // add 2 grams
        if (offset - this.windowLengths[w] >= 0) {
          long prevWord = (wordsForWindowLength[j][offset - this.windowLengths[w]] & mask);
          if (prevWord != 0) {
            long newWord = (prevWord << 32 | word);
            bagOfPatterns[j].bob.putOrAdd(newWord, 1, 1);
          }
        }
      }
    }

    return bagOfPatterns;
  }


  /**
   * Create words and bi-grams for all window lengths
   */
  public BagOfBigrams[] createBagOfPatterns(
      final int[][][] words,
      final TimeSeries[] samples,
      final int wordLength) {
    BagOfBigrams[] bagOfPatterns = new BagOfBigrams[samples.length];

    final byte usedBits = (byte) Words.binlog(this.alphabetSize);
    final long mask = (1L << (usedBits * wordLength)) - 1L;
    int highestBit = Words.binlog(Integer.highestOneBit(WEASELClassifier.MAX_WINDOW_LENGTH))+1;

    // iterate all samples
    // and create a bag of pattern
    for (int j = 0; j < samples.length; j++) {
      bagOfPatterns[j] = new BagOfBigrams(words[0][j].length * 6, samples[j].getLabel());

      // create subsequences
      for (int w = 0; w < this.windowLengths.length; w++) {
        for (int offset = 0; offset < words[w][j].length; offset++) {
          long word = (words[w][j][offset] & mask) << highestBit | (long) w;
          bagOfPatterns[j].bob.putOrAdd(word, 1, 1);

          // add 2 grams
          if (offset - this.windowLengths[w] >= 0) {
            long prevWord = (words[w][j][offset - this.windowLengths[w]] & mask);
            if (prevWord != 0) {
              long newWord = (prevWord << 32 | word);
              bagOfPatterns[j].bob.putOrAdd(newWord, 1, 1);
            }
          }
        }
      }
    }

    return bagOfPatterns;
  }

//  public void trainAnova(final BagOfBigrams[] bob, double p_value) {
//    // compute highest index
//    int length = 0;
//    IntLongHashMap reverseMap = new IntLongHashMap();
//    for (int j = 0; j < bob.length; j++) {
//      for (LongIntCursor word : bob[j].bob) {
//        int index = dict.getWordIndex(word.key);
//        reverseMap.put(index, word.key);
//        length = Math.max(index, length);
//      }
//    }
//
//    // dense double array
//    length = length+1;
//    double[][] data = new double[bob.length][length];
//    for (int i = 0; i < bob.length; i++) {
//      BagOfBigrams bop = bob[i];
//      for (LongIntCursor word : bop.bob) {
//        int index = dict.getWordIndex(word.key);
//        data[i][index] += (double)word.value;
//      }
//    }
//
//    HashMap<Double, ArrayList<double[]>> classes = new HashMap<>();
//    for (int i = 0; i < bob.length; i++) {
//      ArrayList<double[]> allTs = classes.get(bob[i].label);
//      if (allTs == null) {
//        allTs = new ArrayList<>();
//        classes.put(bob[i].label, allTs);
//      }
//      allTs.add(data[i]);
//    }
//
//    double nSamples = bob.length;
//    double nClasses = classes.keySet().size();
//
//    double[] f = SFASupervised.getFoneway(length, classes, nSamples, nClasses);
//
//    // sort by largest f-value
//    @SuppressWarnings("unchecked")
//    List<SFASupervised.Indices<Double>> best = new ArrayList<>(f.length);
//    for (int i = 0; i < f.length; i++) {
//      if (!Double.isNaN(f[i]) && f[i]>0) {
//        best.add(new SFASupervised.Indices<>(i, f[i]));
//      }
//    }
//    Collections.sort(best);
//    best = best.subList(0, (int) Math.min(100, best.size()));
//
//    LongHashSet bestWords = new LongHashSet();
//    for (SFASupervised.Indices<Double> index : best) {
//      bestWords.add(reverseMap.get(index.value.intValue()));
//    }
//
//    for (int j = 0; j < bob.length; j++) {
//      for (LongIntCursor cursor : bob[j].bob) {
//        if (!bestWords.contains(cursor.key)) {
//          bob[j].bob.values[cursor.index] = 0;
//        }
//      }
//    }
//  }


    /**
     * Implementation based on:
     * https://github.com/scikit-learn/scikit-learn/blob/c957249/sklearn/feature_selection/univariate_selection.py#L170
     */
  public void trainChiSquared(final BagOfBigrams[] bob, double chi_limit) {
    // Chi2 Test
    LongIntHashMap featureCount = new LongIntHashMap(bob[0].bob.size());
    LongFloatHashMap classProb = new LongFloatHashMap(10);
    LongObjectHashMap<LongIntHashMap> observed = new LongObjectHashMap<>();

    // count number of samples with this word
    for (BagOfBigrams bagOfPattern : bob) {
      long label = bagOfPattern.label.longValue();
      for (LongIntCursor word : bagOfPattern.bob) {
        if (word.value > 0) {
          featureCount.putOrAdd(word.key, 1, 1);

          int index = -1;
          LongIntHashMap obs = null;
          if ((index = observed.indexOf(label)) > -1) {
            obs = observed.indexGet(index);
          } else {
            obs = new LongIntHashMap();
            observed.put(label, obs);
          }

          // count observations per class for this feature
          obs.putOrAdd(word.key, 1, 1);
        }
      }
    }

    // samples per class
    for (BagOfBigrams bagOfPattern : bob) {
      long label = bagOfPattern.label.longValue();
      classProb.putOrAdd(label, 1, 1);
    }

    // chi-squared: observed minus expected occurrence
    LongHashSet chiSquare = new LongHashSet(featureCount.size());
    ArrayList<PValueKey> pvalues = new ArrayList<PValueKey>(featureCount.size());

    for (LongFloatCursor prob : classProb) {
      prob.value /= bob.length; // (float) frequencies.get(prob.key);

      LongIntHashMap obs = observed.get(prob.key);
      for (LongIntCursor feature : featureCount) {
        double expected = prob.value * feature.value;

        double chi = obs.get(feature.key) - expected;
        double newChi = chi * chi / expected;

        if (newChi > 0 && newChi >= chi_limit
            && !chiSquare.contains(feature.key)) {
          chiSquare.add(feature.key);
          pvalues.add(new PValueKey(newChi, feature.key));
        }
      }
    }

    // limit to 100 (?) features per window size
    int limit = 100;
    if (pvalues.size() > limit) {
      // sort by chi-squared value
      Collections.sort(pvalues, new Comparator<PValueKey>() {
        @Override
        public int compare(PValueKey o1, PValueKey o2) {
          return -Double.compare(o1.pvalue, o2.pvalue);
        }
      });
      // only keep the best featrures (with highest chi-squared pvalue)
      LongHashSet chiSquaredBest = new LongHashSet();
      for (PValueKey key : pvalues.subList(0, Math.min(pvalues.size(), limit))) {
        chiSquaredBest.add(key.key);
      }
      chiSquare = chiSquaredBest;
    }

    for (int j = 0; j < bob.length; j++) {
      for (LongIntCursor cursor : bob[j].bob) {
        if (!chiSquare.contains(cursor.key)) {
          bob[j].bob.values[cursor.index] = 0;
        }
      }
    }
  }

  static class PValueKey {
    public double pvalue;
    public long key;

    public PValueKey(double pvalue, long key) {
      this.pvalue = pvalue;
      this.key = key;
    }

    @Override
    public String toString() {
      return "" + this.pvalue + ":" + this.key;
    }
  }

  /**
   * A dictionary that maps each SFA word to an integer.
   * <p>
   * Condenses the SFA word space.
   */
  public static class Dictionary {
    public LongIntHashMap dict;

    public Dictionary() {
      this.dict = new LongIntHashMap();
    }

    public void reset() {
      this.dict = new LongIntHashMap();
    }

    public int getWordIndex(long word) {
      int index = 0;
      if ((index = this.dict.indexOf(word)) > -1) {
        return this.dict.indexGet(index);
      } else {
        int newWord = this.dict.size() + 1;
        this.dict.put(word, newWord);
        return newWord;
      }
    }

    public int size() {
      return this.dict.size();
    }

    public void filterChiSquared(final BagOfBigrams[] bagOfPatterns) {
      for (int j = 0; j < bagOfPatterns.length; j++) {
        LongIntHashMap oldMap = bagOfPatterns[j].bob;
        bagOfPatterns[j].bob = new LongIntHashMap();
        for (LongIntCursor word : oldMap) {
          if (this.dict.containsKey(word.key) && word.value > 0) {
            bagOfPatterns[j].bob.put(word.key, word.value);
          }
        }
      }
    }
  }
}
