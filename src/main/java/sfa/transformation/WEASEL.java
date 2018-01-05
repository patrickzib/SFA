// Copyright (c) 2017 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.transformation;

import com.carrotsearch.hppc.*;
import com.carrotsearch.hppc.cursors.IntIntCursor;
import com.carrotsearch.hppc.cursors.LongFloatCursor;
import sfa.classification.Classifier;
import sfa.classification.Classifier.Words;
import sfa.classification.ParallelFor;
import sfa.timeseries.TimeSeries;

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
    public IntIntHashMap bob;
    public Double label;

    public BagOfBigrams(int size, Double label) {
      this.bob = new IntIntHashMap(size);
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
  private int[][] createWords(final TimeSeries[] samples, final int index) {

    // SFA quantization
    if (this.signature[index] == null) {
      this.signature[index] = new SFASupervised();
      this.signature[index].fitWindowing(
          samples, this.windowLengths[index], this.maxF, this.alphabetSize, this.normMean, this.lowerBounding);
    }

    // create words
    final int[][] words = new int[samples.length][];
    for (int i = 0; i < samples.length; i++) {
      words[i] = this.signature[index].transformWindowingInt(samples[i], this.maxF);
    }

    return words;
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

    // FIXME
    //    final long mask = (usedBits << wordLength) - 1L;
    final long mask = (1L << (usedBits * wordLength)) - 1L;
    int highestBit = Words.binlog(Integer.highestOneBit(Classifier.MAX_WINDOW_LENGTH))+1;

    // iterate all samples
    // and create a bag of pattern
    for (int j = 0; j < samples.length; j++) {
      bagOfPatterns[j] = new BagOfBigrams(words[0][j].length * 6, samples[j].getLabel());

      // create subsequences
      for (int w = 0; w < this.windowLengths.length; w++) {
        for (int offset = 0; offset < words[w][j].length; offset++) {
          int word = this.dict.getWord((words[w][j][offset] & mask) << highestBit | (long) w);
          bagOfPatterns[j].bob.putOrAdd(word, 1, 1);

          // add 2 grams
          if (offset - this.windowLengths[w] >= 0) {
            long prevWord = this.dict.getWord((words[w][j][offset - this.windowLengths[w]] & mask) << highestBit | (long) w);
            int newWord = this.dict.getWord((prevWord << 32 | word ) << highestBit);
            bagOfPatterns[j].bob.putOrAdd(newWord, 1, 1);
          }
        }
      }
    }

    return bagOfPatterns;
  }

  /**
   * Implementation based on:
   * https://github.com/scikit-learn/scikit-learn/blob/c957249/sklearn/feature_selection/univariate_selection.py#L170
   */
  public void filterChiSquared(final BagOfBigrams[] bob, double chi_limit) {
    // Chi2 Test
    IntIntHashMap featureCount = new IntIntHashMap(bob[0].bob.size());
    LongFloatHashMap classProb = new LongFloatHashMap(10);
    LongIntHashMap observed = new LongIntHashMap(bob[0].bob.size());

    // count number of samples with this word
    for (BagOfBigrams bagOfPattern : bob) {
      long label = bagOfPattern.label.longValue();
      for (IntIntCursor word : bagOfPattern.bob) {
        if (word.value > 0) {
          featureCount.putOrAdd(word.key, 1, 1);
          long key = label << 32 | word.key;
          observed.putOrAdd(key, 1, 1);
        }
      }
    }

    // samples per class
    for (BagOfBigrams bagOfPattern : bob) {
      long label = bagOfPattern.label.longValue();
      classProb.putOrAdd(label, 1, 1);
    }

    // chi-squared: observed minus expected occurrence
    IntHashSet chiSquare = new IntHashSet(featureCount.size());
    for (LongFloatCursor prob : classProb) {
      prob.value /= bob.length; // (float) frequencies.get(prob.key);

      for (IntIntCursor feature : featureCount) {
        long key = prob.key << 32 | feature.key;
        float expected = prob.value * feature.value;

        float chi = observed.get(key) - expected;
        float newChi = chi * chi / expected;
        if (newChi >= chi_limit
            && !chiSquare.contains(feature.key)) {
          chiSquare.add(feature.key);
        }
      }
    }

    // best elements above limit
    for (int j = 0; j < bob.length; j++) {
      for (IntIntCursor cursor : bob[j].bob) {
        if (!chiSquare.contains(cursor.key)) {
          bob[j].bob.values[cursor.index] = 0;
        }
      }
    }

    // chi-squared reduces keys substantially => remap
    this.dict.remap(bob);
  }

  /**
   * A dictionary that maps each SFA word to an integer.
   * <p>
   * Condenses the SFA word space.
   */
  public static class Dictionary {
    LongIntHashMap dict;
    LongIntHashMap dictChi;

    public Dictionary() {
      this.dict = new LongIntHashMap();
      this.dictChi = new LongIntHashMap();
    }

    public void reset() {
      this.dict = new LongIntHashMap();
      this.dictChi = new LongIntHashMap();
    }

    public int getWord(long word) {
      int index = 0;
      if ((index = this.dict.indexOf(word)) > -1) {
        return this.dict.indexGet(index);
      } else {
        int newWord = this.dict.size() + 1;
        this.dict.put(word, newWord);
        return newWord;
      }
    }

    public int getWordChi(long word) {
      int index = 0;
      if ((index = this.dictChi.indexOf(word)) > -1) {
        return this.dictChi.indexGet(index);
      } else {
        int newWord = this.dictChi.size() + 1;
        this.dictChi.put(word, newWord);
        return newWord;
      }
    }

    public int size() {
      if (!this.dictChi.isEmpty()) {
        return this.dictChi.size();
      } else {
        return this.dict.size();
      }
    }

    public void remap(final BagOfBigrams[] bagOfPatterns) {
      for (int j = 0; j < bagOfPatterns.length; j++) {
        IntIntHashMap oldMap = bagOfPatterns[j].bob;
        bagOfPatterns[j].bob = new IntIntHashMap();
        for (IntIntCursor word : oldMap) {
          if (word.value > 0) {
            bagOfPatterns[j].bob.put(getWordChi(word.key), word.value);
          }
        }
      }
    }
  }
}
