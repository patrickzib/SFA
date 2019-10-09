// Copyright (c) 2017 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.transformation;

import com.carrotsearch.hppc.*;
import com.carrotsearch.hppc.cursors.*;
import org.apache.commons.math3.distribution.ChiSquaredDistribution;
import org.apache.commons.math3.distribution.FDistribution;
import sfa.classification.Classifier.Words;
import sfa.classification.ParallelFor;
import sfa.classification.WEASELClassifier;
import sfa.timeseries.TimeSeries;

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
    int processed = ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        for (int w = 0; w < WEASEL.this.windowLengths.length; w++) {
          if (w % BLOCKS == id) {
            words[w] = createWords(samples, w);
            processed.incrementAndGet();
          }
        }
     }
    });
    if (processed < WEASEL.this.windowLengths.length) {
      System.err.println("Processed: " + processed + " < " + WEASEL.this.windowLengths.length + " windowlengths.");
    }
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

//        // add 2 grams
//        if (offset - this.windowLengths[w] >= 0) {
//          long prevWord = (wordsForWindowLength[j][offset - this.windowLengths[w]] & mask);
//          if (prevWord != 0) {
//            long newWord = (prevWord << 32 | word);
//            bagOfPatterns[j].bob.putOrAdd(newWord, 1, 1);
//          }
//        }
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
      bagOfPatterns[j] = new BagOfBigrams(words[0][j].length * 2, samples[j].getLabel());

      // create subsequences
      for (int w = 0; w < this.windowLengths.length; w++) {
        for (int offset = 0; offset < words[w][j].length; offset++) {
          long word = (words[w][j][offset] & mask) << highestBit | (long) w;
          bagOfPatterns[j].bob.putOrAdd(word, 1, 1);

//          // add 2 grams
//          if (offset - this.windowLengths[w] >= 0) {
//            long prevWord = (words[w][j][offset - this.windowLengths[w]] & mask);
//            if (prevWord != 0) {
//              long newWord = (prevWord << 32 | word);
//              bagOfPatterns[j].bob.putOrAdd(newWord, 1, 1);
//            }
//          }
        }
      }
    }

    return bagOfPatterns;
  }

  /**
   * Implementation based on:
   * https://github.com/scikit-learn/scikit-learn/blob/c957249/sklearn/feature_selection/univariate_selection.py#L170
   */
  public void trainChiSquared(final BagOfBigrams[] bob, double p_limit) {
    // Chi2 Test
    LongIntHashMap featureCount = new LongIntHashMap(bob[0].bob.size());
    DoubleIntHashMap classProb = new DoubleIntHashMap(10);
    DoubleObjectHashMap<LongIntHashMap> observed = new DoubleObjectHashMap<>();

    // count number of samples with this word
    for (BagOfBigrams bagOfPattern : bob) {
      double label = bagOfPattern.label;

      int index = -1;
      LongIntHashMap obs = null;
      if ((index = observed.indexOf(label)) > -1) {
        obs = observed.indexGet(index);
      } else {
        obs = new LongIntHashMap();
        observed.put(label, obs);
      }

      for (LongIntCursor word : bagOfPattern.bob) {
        if (word.value > 0) {
          featureCount.putOrAdd(word.key, 1,1); //word.value, word.value);

          // count observations per class for this feature
          obs.putOrAdd(word.key, 1,1); //word.value, word.value);
        }
      }
    }

    // samples per class
    for (BagOfBigrams bagOfPattern : bob) {
      double label = bagOfPattern.label;
      classProb.putOrAdd(label, 1, 1);
    }

    // p_value-squared: observed minus expected occurrence
    LongDoubleHashMap chiSquareSum = new LongDoubleHashMap(featureCount.size());

    for (DoubleIntCursor prob : classProb) {
      double p = ((double)prob.value) / bob.length;
      LongIntHashMap obs = observed.get(prob.key);

      for (LongIntCursor feature : featureCount) {
        double expected = p * feature.value;

        double chi = obs.get(feature.key) - expected;
        double newChi = chi * chi / expected;

        if (newChi > 0) {
          // build the sum among p_value-values of all classes
          chiSquareSum.putOrAdd(feature.key, newChi, newChi);
        }
      }
    }

    LongHashSet chiSquare = new LongHashSet(featureCount.size());
    ArrayList<PValueKey> values = new ArrayList<PValueKey>(featureCount.size());
    final ChiSquaredDistribution distribution
        = new ChiSquaredDistribution(null, classProb.keys().size()-1);

    for (LongDoubleCursor feature : chiSquareSum) {
      double newChi = feature.value;
      double pvalue = 1.0 - distribution.cumulativeProbability(newChi);
      if (pvalue <= p_limit) {
        //System.out.println(newChi + " " + pvalue);
        chiSquare.add(feature.key);
        values.add(new PValueKey(pvalue, feature.key));
      }
    }

    // limit number of features per window size to avoid excessive features
    int limit = 100;
    if (values.size() > limit) {
      // sort by p_value-squared value
      Collections.sort(values, new Comparator<PValueKey>() {
        @Override
        public int compare(PValueKey o1, PValueKey o2) {
          int comp = Double.compare(o1.pvalue, o2.pvalue);
          if (comp != 0) { // tie breaker
            return comp;
          }
          return Long.compare(o1.key, o2.key);
        }
      });

      chiSquare.clear();


      // use 100 unigrams and 100 bigrams
      int countUnigram = 0;
      int countBigram = 0;
      for (int i = 0; i < values.size(); i++) {
        // bigram?
        long val = values.get(i).key;
        if (val > (1l << 32) && countBigram < limit) {
          chiSquare.add(val);
          countBigram++;
        }
        // unigram?
        else if (val < (1l << 32) && countUnigram < limit){
          chiSquare.add(val);
          countUnigram++;
        }

        if (countUnigram >= limit && countBigram >= limit) {
          break;
        }
      }
    }

    // remove values
    for (int j = 0; j < bob.length; j++) {
      LongIntHashMap oldMap = bob[j].bob;
      bob[j].bob = new LongIntHashMap();
      for (LongIntCursor cursor : oldMap) {
        if (chiSquare.contains(cursor.key)) {
          bob[j].bob.put(cursor.key, cursor.value);
        }
      }
      oldMap.clear();
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
    public LongIntHashMap dict; /* stores the ID of each word */
    public ArrayList<Long> inverseDict; /* map IDs back to words */

    public Dictionary() {
      this.dict = new LongIntHashMap();
      this.inverseDict = new ArrayList<>();
      this.inverseDict.add(0l); // add dummy
    }

    public void reset() {
      this.dict = new LongIntHashMap();
      this.inverseDict = new ArrayList<>(1);
      this.inverseDict.add(0l); // add dummy
    }

    public int getWordIndex(long word, boolean add) {
      int index = 0;
      if ((index = this.dict.indexOf(word)) > -1) {
        return this.dict.indexGet(index);
      } else if (add) {
        int newWord = this.dict.size() + 1;
        this.dict.put(word, newWord);
        inverseDict.add(/*newWord,*/ word);
//        if (inverseDict.get(newWord) != word) {
//          System.out.println("Error");
//        }
        return newWord;
      } else {
        return -1;
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
