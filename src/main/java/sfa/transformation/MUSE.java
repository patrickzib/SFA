// Copyright (c) 2017 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.transformation;

import com.carrotsearch.hppc.*;
import com.carrotsearch.hppc.cursors.IntIntCursor;
import com.carrotsearch.hppc.cursors.LongDoubleCursor;

import com.carrotsearch.hppc.cursors.ObjectIntCursor;
import sfa.classification.Classifier;
import sfa.classification.MUSEClassifier;
import sfa.classification.ParallelFor;
import sfa.timeseries.MultiVariateTimeSeries;
import sfa.timeseries.TimeSeries;

import java.util.ArrayList;
import java.util.HashMap;
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
  public SFA[][] signature;
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
    this.signature = new SFA[windowLengths.length][];
    this.histogramType = histogramType;
  }

  /**
   * The MUSE model: a histogram of SFA word and bi-gram frequencies
   */
  public static class BagOfBigrams {
    public ObjectIntHashMap<MuseWord> bob;
    public Double label;

    public BagOfBigrams(int size, Double label) {
      this.bob = new ObjectIntHashMap<MuseWord>(size);
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
  public int[/*sample size*/][] createWords(final MultiVariateTimeSeries[] mtsSamples, final int index) {

    // SFA quantization
    if (this.signature[index] == null) {
      this.signature[index] = new SFA[mtsSamples[0].getDimensions()]; 
      for (int i = 0; i < this.signature[index].length; i++) {
        this.signature[index][i] = new SFA(this.histogramType, false); // TODO true?
        this.signature[index][i].fitWindowing(
            mtsSamples, this.windowLengths[index], this.maxF, this.alphabetSize, this.normMean, this.lowerBounding, i);
      }
    }

    // create words
    final int[][] words = new int[mtsSamples.length * mtsSamples[0].getDimensions()][];
    int pos = 0;
    for (MultiVariateTimeSeries mts : mtsSamples) {
      int i = 0;      
      for (TimeSeries timeSeries : mts.timeSeries) {
        if (timeSeries.getLength() >= this.windowLengths[index]) {
          words[pos] = this.signature[index][i].transformWindowingInt(timeSeries, this.maxF);
        } else {
          words[pos] = new int[]{};
        }
        i++;
        pos++;
      }
    }

    return words;
  }

  public class MuseWord {
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
              //int dict = this.dict.getWord(word);
              bop.bob.putOrAdd(word, 1, 1);

              // add bigrams
              if (this.windowLengths[this.windowLengths.length-1] < 200 // avoid for large datasets
                  && MUSEClassifier.BIGRAMS 
                  && (offset - this.windowLengths[w] >= 0)) {
                MuseWord bigram = new MuseWord(w, dim,
                    (words[w][j + dim][offset - this.windowLengths[w]] & mask),
                    words[w][j + dim][offset] & mask);
                //int newWord = this.dict.getWord(bigram);
                bop.bob.putOrAdd(bigram, 1, 1);
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
   * Create words and bi-grams for all window lengths
   */
  public BagOfBigrams[] createBagOfPatterns(
      final int[][] wordsForWindowLength,
      final MultiVariateTimeSeries[] samples,
      final int w,    // index of used windowSize
      final int dimensionality,
      final int wordLength) {
    List<BagOfBigrams> bagOfPatterns = new ArrayList<BagOfBigrams>(
        samples[0].getDimensions() * samples.length);

    final byte usedBits = (byte) Classifier.Words.binlog(this.alphabetSize);
    final int mask = (1 << (usedBits * wordLength)) - 1;

    // iterate all samples and create a muse model for each
    for (int i = 0, j = 0; i < samples.length; i++, j += dimensionality) {
      BagOfBigrams bop = new BagOfBigrams(100, samples[i].getLabel());

      // create subsequences
      if (this.windowLengths[w] >= wordLength) {
        for (int dim = 0; dim < dimensionality; dim++) {
          for (int offset = 0; offset < wordsForWindowLength[j + dim].length; offset++) {
            MuseWord word = new MuseWord(w, dim, wordsForWindowLength[j + dim][offset] & mask, 0);
            //int dict = this.dict.getWord(word);
            bop.bob.putOrAdd(word, 1, 1);

            // add bigrams
            if (this.windowLengths[this.windowLengths.length-1] < 200 // avoid for too large datasets
                && MUSEClassifier.BIGRAMS
                && (offset - this.windowLengths[w] >= 0)) {
              MuseWord bigram = new MuseWord(w, dim,
                  (wordsForWindowLength[j + dim][offset - this.windowLengths[w]] & mask),
                  wordsForWindowLength[j + dim][offset] & mask);
              //int newWord = this.dict.getWord(bigram);
              bop.bob.putOrAdd(bigram, 1, 1);
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
    ObjectIntHashMap<MuseWord> featureCount = new ObjectIntHashMap<>(bob[0].bob.size());
    LongDoubleHashMap classProb = new LongDoubleHashMap(10);
    LongObjectHashMap<ObjectIntHashMap<MuseWord>> observed = new LongObjectHashMap<>(bob[0].bob.size());

    // count number of samples with this word
    for (BagOfBigrams bagOfPattern : bob) {
      long label = bagOfPattern.label.longValue();
      if (!observed.containsKey(label)) {
        observed.put(label, new ObjectIntHashMap<>());
      }
      for (ObjectIntCursor<MuseWord> word : bagOfPattern.bob) {
        if (word.value > 0) {
          featureCount.putOrAdd(word.key, 1, 1);
          observed.get(label).putOrAdd(word.key, 1, 1);
        }
      }
    }

    // samples per class
    for (BagOfBigrams bagOfPattern : bob) {
      long label = bagOfPattern.label.longValue();
      classProb.putOrAdd(label, 1, 1);
    }

    // p_value-squared: observed minus expected occurrence
    ObjectHashSet<MuseWord> chiSquare = new ObjectHashSet<>(featureCount.size());
    for (LongDoubleCursor classLabel : classProb) {
      classLabel.value /= (double) bob.length;
      if (observed.get(classLabel.key) != null) {
        ObjectIntHashMap<MuseWord> observe = observed.get(classLabel.key);
        for (ObjectIntCursor<MuseWord> feature : featureCount) {
          double expected = classLabel.value * feature.value;
          double chi = observe.get(feature.key) - expected;
          double newChi = chi * chi / expected;
          if (newChi >= chi_limit
              && !chiSquare.contains(feature.key)) {
            chiSquare.add(feature.key);
          }
        }
      }
    }

    // best elements above limit
    for (int j = 0; j < bob.length; j++) {
      for (ObjectIntCursor<MuseWord> cursor : bob[j].bob) {
        if (!chiSquare.contains(cursor.key)) {
          bob[j].bob.values[cursor.index] = 0;
        }
      }
    }
  }

  /**
   * A dictionary that maps each SFA word to an integer.
   * <p>
   * Condenses the SFA word space.
   */
  public static class Dictionary {
    public ObjectIntHashMap<MuseWord> dictChi;

    public Dictionary() {
      this.dictChi = new ObjectIntHashMap<MuseWord>();
    }

    public void reset() {
      this.dictChi = new ObjectIntHashMap<MuseWord>();
    }

    public int getWordChi(MuseWord word) {
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
      return this.dictChi.size();
    }

    public void filterChiSquared(final BagOfBigrams[] bagOfPatterns) {
      for (int j = 0; j < bagOfPatterns.length; j++) {
        ObjectIntHashMap<MuseWord> oldMap = bagOfPatterns[j].bob;
        bagOfPatterns[j].bob = new ObjectIntHashMap<MuseWord>();
        for (ObjectIntCursor<MuseWord> word : oldMap) {
          if (this.dictChi.containsKey(word.key) && word.value > 0) {
            bagOfPatterns[j].bob.put(word.key, word.value);
          }
        }
      }
    }
  }
}
