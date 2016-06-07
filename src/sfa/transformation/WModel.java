// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.transformation;

import java.util.concurrent.atomic.AtomicInteger;

import sfa.classification.Classifier.Words;
import sfa.classification.ParallelFor;
import sfa.timeseries.TimeSeries;
import sfa.transformation.SFA.HistogramType;

import com.carrotsearch.hppc.IntFloatOpenHashMap;
import com.carrotsearch.hppc.IntIntOpenHashMap;
import com.carrotsearch.hppc.LongFloatOpenHashMap;
import com.carrotsearch.hppc.LongIntOpenHashMap;
import com.carrotsearch.hppc.cursors.IntIntCursor;
import com.carrotsearch.hppc.cursors.LongFloatCursor;

/**
 * The W-Model.
 *
 * @author bzcschae
 *
 */
public class WModel {

  public int alphabetSize;
  public int maxF;

  public int[] windowLengths;
  public boolean norm;
  public boolean normMean;
  public boolean lowerBounding;
  public SFA[] signature;
  public Dictionary dict;

  public final static int BLOCKS;

  static {
    Runtime runtime = Runtime.getRuntime();
    if (runtime.availableProcessors() <= 4) {
      BLOCKS = 8;
    }
    else {
      BLOCKS = runtime.availableProcessors();
    }
    
//    BLOCKS = 1;
  }

  public WModel(
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
   * The W-model: a histogram of SFA word and bi-gram frequencies
   */
  public static class BagOfBigrams {
    public IntIntOpenHashMap bob;
    public String label;

    public BagOfBigrams(int size, String label) {
      this.bob = new IntIntOpenHashMap(size);
      this.label = label;
    }
  }

  /**
   * Create SFA words and bigrams for all samples
   * @param samples
   * @return
   */
  public int[][][] createWords(final TimeSeries[] samples) {
    // create bag of words for each window length
    final int[][][] words = new int[this.windowLengths.length][samples.length][];
    ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        for (int w = 0; w < WModel.this.windowLengths.length; w++) {
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
   * @param samples
   * @return
   */
  private int[][] createWords(final TimeSeries[] samples, final int index) {

    // SFA quantization
    if (this.signature[index] == null) {
      this.signature[index] = new SFASupervised();
      this.signature[index].fitWindowing(samples, this.windowLengths[index], this.maxF, this.alphabetSize, this.normMean, this.lowerBounding);
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

    final byte usedBits = (byte)Words.binlog(this.alphabetSize);
    final int count = usedBits*wordLength;
    final long mask = (1l << (count)) - 1l;

    // iterate all samples
    // and create a bag of pattern
    for (int j = 0; j < samples.length; j++) {
      bagOfPatterns[j] = new BagOfBigrams(words[0][j].length*6, samples[j].getLabel());

      // create subsequences
      for (int w = 0; w < this.windowLengths.length; w++) {
        final short factor = 1;
        for (int offset = 0; offset < words[w][j].length; offset++) {
          int word = this.dict.getWord( (long)w << 52 | (words[w][j][offset] & mask));
          bagOfPatterns[j].bob.putOrAdd(word, factor, factor);

          // add 2 grams
          if (offset-this.windowLengths[w] >= 0) {
            long prevWord = this.dict.getWord( (long)w << 52 | (words[w][j][offset-this.windowLengths[w]] & mask));
            int newWord = this.dict.getWord( (long)w << 52 |  prevWord << 26 | (long)word);
            bagOfPatterns[j].bob.putOrAdd(newWord, factor, factor);
          }
        }
      }
    }
    return bagOfPatterns;
  }

  /**
   *
   * Implementation based on:
   *    https://github.com/scikit-learn/scikit-learn/blob/c957249/sklearn/feature_selection/univariate_selection.py#L170
   *
   */
  public void filterChiSquared(final BagOfBigrams[] bob, double chi_limit) {
    // class frequencies
    LongIntOpenHashMap classFrequencies  = new LongIntOpenHashMap();
    for (BagOfBigrams ts : bob) {
      long label = Double.valueOf(ts.label).longValue();
      classFrequencies.putOrAdd(label, 1, 1);
    }

    // Chi2 Test
    IntIntOpenHashMap featureCount = new IntIntOpenHashMap(bob[0].bob.size());
    LongFloatOpenHashMap classProb = new LongFloatOpenHashMap(10);
    LongIntOpenHashMap observed = new LongIntOpenHashMap(bob[0].bob.size());
    IntFloatOpenHashMap chiSquare = new IntFloatOpenHashMap(bob[0].bob.size());

    // count number of samples with this word
    for (BagOfBigrams bagOfPattern : bob) {
      long label = Double.valueOf(bagOfPattern.label).longValue();
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
      long label = Double.valueOf(bagOfPattern.label).longValue();
      classProb.putOrAdd(label, 1, 1);
    }

    // chi square: observed minus expected occurence
    for (LongFloatCursor prob : classProb) {
      prob.value /= (float) bob.length; // (float) frequencies.get(prob.key);

      for (IntIntCursor feature : featureCount) {
        long key = prob.key << 32 | feature.key;
        float expected = prob.value * feature.value;

        float chi = observed.get(key) - expected;
        float newChi = chi*chi / expected;
        if (newChi >= chi_limit
            && newChi > chiSquare.get(feature.key)) {
          chiSquare.put(feature.key, newChi);
        }
      }
    }

    // best elements above limit
    for (int j = 0; j < bob.length; j++) {
      for (IntIntCursor cursor : bob[j].bob) {
        if (chiSquare.get(cursor.key) < chi_limit) {
          bob[j].bob.values[cursor.index] = 0;
        }
      }
    }

    // chi square reduces keys substantially => remap
    this.dict.remap(bob);
  }

  /**
   * A dictionary that maps each SFA word to an integer.
   *
   * Condenses the SFA word space.
   */
  public static class Dictionary {
    LongIntOpenHashMap dict;
    LongIntOpenHashMap dictChi;

    public Dictionary() {
      this.dict = new LongIntOpenHashMap();
      this.dictChi = new LongIntOpenHashMap();
    }

    public void reset() {
      this.dict = new LongIntOpenHashMap();
      this.dictChi = new LongIntOpenHashMap();
    }

    public int getWord(long word) {
      if (this.dict.containsKey(word)) {
        word = this.dict.lget();
      }
      else {
        int newWord = this.dict.size()+1;
        this.dict.put(word, newWord);
        word = newWord;
      }
      return (int)word;
    }

    public int getWordChi(long word) {
      if (this.dictChi.containsKey(word)) {
        return this.dictChi.lget();
      }
      else {
        int newWord = this.dictChi.size()+1;
        this.dictChi.put(word, newWord);
        return newWord;
      }
    }

    public int size() {
      if (!this.dictChi.isEmpty()) {
        return this.dictChi.size();
      }
      else {
        return this.dict.size();
      }
    }

    public void remap(final BagOfBigrams[] bagOfPatterns) {
      for (int j = 0; j < bagOfPatterns.length; j++) {
        IntIntOpenHashMap oldMap = bagOfPatterns[j].bob;
        bagOfPatterns[j].bob = new IntIntOpenHashMap(oldMap.size());
        for (IntIntCursor word : oldMap) {
          if (word.value > 0) {
            bagOfPatterns[j].bob.put(getWordChi(word.key), word.value);
          }
        }
      }
    }
  }
}
