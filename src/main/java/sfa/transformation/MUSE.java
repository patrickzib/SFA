// Copyright (c) 2017 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.transformation;

import com.carrotsearch.hppc.*;
import com.carrotsearch.hppc.cursors.IntIntCursor;
import com.carrotsearch.hppc.cursors.LongDoubleCursor;
import sfa.classification.Classifier;
import sfa.classification.MUSEClassifier;
import sfa.classification.ParallelFor;
import sfa.timeseries.MultiVariateTimeSeries;
import sfa.timeseries.TimeSeries;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The WEASEL+MUSE-Model as published in
 * <p>
 * Schäfer, P., Leser, U.: Multivariate Time Series Classification
 * with WEASEL+MUSE. arXiv 2017
 * http://arxiv.org/abs/1711.11343
 */
public class MUSE {

  public int alphabetSize;
  public int maxF;
  public SFA.HistogramType histogramType = null;

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
  }

  /**
   * Create a WEASEL+MUSE model.
   *
   * @param maxF          Length of the SFA words
   * @param maxS          alphabet size
   * @param histogramType histogram types (EQUI-Depth and/or EQUI-Frequency) to use
   * @param windowLengths the set of window lengths to use for extracting SFA words from
   *                      time series.
   * @param normMean      set to true, if mean should be set to 0 for a window
   * @param lowerBounding set to true, if the Fourier transform should be normed (typically
   *                      used to lower bound / mimic Euclidean distance).
   */
  public MUSE(
      int maxF,
      int maxS,
      SFA.HistogramType histogramType,
      int[] windowLengths,
      boolean normMean,
      boolean lowerBounding) {
    this.maxF = maxF + maxF % 2; // even number
    this.alphabetSize = maxS;
    this.windowLengths = windowLengths;
    this.normMean = normMean;
    this.lowerBounding = lowerBounding;
    this.dict = new Dictionary();
    this.signature = new SFA[windowLengths.length];
    this.histogramType = histogramType;
  }

  /**
   * The MUSE model: a histogram of SFA word and bi-gram frequencies
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
  public int[][][] createWords(final MultiVariateTimeSeries[] samples) {
    // create bag of words for each window length
    final int[][][] words = new int[this.windowLengths.length][samples.length][];
    ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        for (int w = 0; w < MUSE.this.windowLengths.length; w++) {
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
   * @param mtsSamples
   * @return
   */
  private int[/*window size*/][] createWords(final MultiVariateTimeSeries[] mtsSamples, final int index) {

    // SFA quantization
    if (this.signature[index] == null) {
      this.signature[index] = new SFA(this.histogramType, true);
      this.signature[index].fitWindowing(
          mtsSamples, this.windowLengths[index], this.maxF, this.alphabetSize, this.normMean, this.lowerBounding);
    }

    // create words
    final int[][] words = new int[mtsSamples.length * mtsSamples[0].getDimensions()][];
    int pos = 0;
    for (MultiVariateTimeSeries mts : mtsSamples) {
      for (TimeSeries timeSeries : mts.timeSeries) {
        if (timeSeries.getLength() >= this.windowLengths[index]) {
          words[pos] = this.signature[index].transformWindowingInt(timeSeries, this.maxF);
        } else {
          words[pos] = new int[]{};
        }
        pos++;
      }
    }

    return words;
  }

  class MuseWord {
    int w = 0;
    int dim = 0;
    int word = 0;
    int word2 = 0;
    public MuseWord(int w, int dim, int word, int word2) {
      this.w = w;
      this.dim = dim;
      this.word = word;
      this.word2 = word2;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) return true;
      MuseWord museWord = (MuseWord) o;
      return w == museWord.w &&
          dim == museWord.dim &&
          word == museWord.word &&
          word2 == museWord.word2;
    }

    @Override
    public int hashCode() {
      int result = 1;
      result = 31 * result + Integer.hashCode(word);
      result = 31 * result + Integer.hashCode(word2);
      result = 31 * result + Integer.hashCode(w);
      result = 31 * result + Integer.hashCode(dim);
      return result;
    }
  }


  /**
   * Create words and bi-grams for all window lengths
   */
  public BagOfBigrams[] createBagOfPatterns(
      final int[][][] words,
      final MultiVariateTimeSeries[] samples,
      final int dimensionality,
      final int wordLength) {
    List<BagOfBigrams> bagOfPatterns = new ArrayList<BagOfBigrams>(
        samples[0].getDimensions() * samples.length);

    final byte usedBits = (byte) Classifier.Words.binlog(this.alphabetSize);
//    final long mask = (usedBits << wordLength) - 1l;
    final int mask = (1 << (usedBits * wordLength)) - 1;

    // iterate all samples and create a muse model for each
    for (int i = 0, j = 0; i < samples.length; i++, j += dimensionality) {
      BagOfBigrams bop = new BagOfBigrams(100, samples[i].getLabel());

      // create subsequences
      for (int w = 0; w < this.windowLengths.length; w++) {
        if (this.windowLengths[w] >= wordLength) {
          for (int dim = 0; dim < dimensionality; dim++) {
            for (int offset = 0; offset < words[w][j + dim].length; offset++) {
              MuseWord word = new MuseWord(w, dim, words[w][j + dim][offset] & mask, 0);
              int dict = this.dict.getWord(word);
              bop.bob.putOrAdd(dict, 1, 1);

              // add bigrams
              if (MUSEClassifier.BIGRAMS && (offset - this.windowLengths[w] >= 0)) {
                MuseWord bigram = new MuseWord(w, dim,
                    (words[w][j + dim][offset - this.windowLengths[w]] & mask),
                    words[w][j + dim][offset] & mask);
                int newWord = this.dict.getWord(bigram);
                bop.bob.putOrAdd(newWord, 1, 1);
              }
            }
          }
        }
      }
      bagOfPatterns.add(bop);
    }
    return bagOfPatterns.toArray(new BagOfBigrams[]{});
  }

  /**
   * Implementation based on:
   * https://github.com/scikit-learn/scikit-learn/blob/c957249/sklearn/feature_selection/univariate_selection.py#L170
   */
  public void filterChiSquared(final BagOfBigrams[] bob, double chi_limit) {
    // Chi^2 Test
    IntIntHashMap featureCount = new IntIntHashMap(bob[0].bob.size());
    LongDoubleHashMap classProb = new LongDoubleHashMap(10);
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
    for (LongDoubleCursor classLabel : classProb) {
      classLabel.value /= (double) bob.length; // (double) frequencies.get(classLabel.key);

      for (IntIntCursor feature : featureCount) {
        long key = classLabel.key << 32 | feature.key;
        double expected = classLabel.value * feature.value;

        double chi = observed.get(key) - expected;
        double newChi = chi * chi / expected;
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
    ObjectIntHashMap<MuseWord> dict;
    IntIntHashMap dictChi;

    public Dictionary() {
      this.dict = new ObjectIntHashMap<MuseWord>();
      this.dictChi = new IntIntHashMap();
    }

    public void reset() {
      this.dict = new ObjectIntHashMap<MuseWord>();
      this.dictChi = new IntIntHashMap();
    }

    public int getWord(MuseWord word) {
      int index = 0;
      int newWord = -1;
      if ((index = this.dict.indexOf(word)) > -1) {
        newWord = this.dict.indexGet(index);
      } else {
        newWord = this.dict.size() + 1;
        this.dict.put(word, newWord);
      }
      return newWord;
    }

    public int getWordChi(int word) {
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
        bagOfPatterns[j].bob = new IntIntHashMap(oldMap.size());
        for (IntIntCursor word : oldMap) {
          if (word.value > 0) {
            bagOfPatterns[j].bob.put(getWordChi(word.key), word.value);
          }
        }
      }
    }
  }
}
