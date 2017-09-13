// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.transformation;

import java.util.concurrent.atomic.AtomicInteger;

import sfa.classification.Classifier.Words;
import sfa.classification.ParallelFor;
import sfa.timeseries.TimeSeries;
import sfa.transformation.SFA.HistogramType;

import com.carrotsearch.hppc.IntIntHashMap;

/**
 * The Bag-of-SFA-Symbols model as published in
 * Schäfer, P.: The boss is concerned with time series classification
 * in the presence of noise. DMKD 29(6) (2015) 1505–1530
 *
 * @author bzcschae
 */
public class BOSSModel {

  public int symbols;
  public int maxF;
  public int windowLength;
  public boolean normMean;
  public SFA signature;
  public final static int BLOCKS = 8;

  /**
   * Create a BOSS model.
   *
   * @param maxF         length of the SFA words
   * @param maxS         alphabet size
   * @param windowLength sub-sequence (window) length used for extracting SFA words from
   *                     time series.
   * @param normMean     set to true, if mean should be set to 0 for a window
   */
  public BOSSModel(int maxF, int maxS, int windowLength, boolean normMean) {
    this.maxF = maxF;
    this.symbols = maxS;
    this.windowLength = windowLength;
    this.normMean = normMean;
  }

  /**
   * The BOSS model: a histogram of SFA word frequencies
   */
  public static class BagOfPattern {
    public IntIntHashMap bag;
    public String label;

    public BagOfPattern(int size, String label) {
      this.bag = new IntIntHashMap(size);
      this.label = label;
    }
  }

  /**
   * Create SFA words for all samples
   *
   * @param samples the time series to be transformed
   * @return returns an array of words for each time series
   */
  public int[][] createWords(final TimeSeries[] samples) {

    final int[][] words = new int[samples.length][];

    if (this.signature == null) {
      this.signature = new SFA(HistogramType.EQUI_DEPTH);
      this.signature.fitWindowing(samples, this.windowLength, this.maxF, this.symbols, this.normMean, true);
    }

    // create sliding windows
    ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        for (int i = 0; i < samples.length; i++) {
          if (i % BLOCKS == id) {
            short[][] sfaWords = BOSSModel.this.signature.transformWindowing(samples[i]);
            words[i] = new int[sfaWords.length];
            for (int j = 0; j < sfaWords.length; j++) {
              words[i][j] = (int) Words.createWord(sfaWords[j], BOSSModel.this.maxF, (byte) Words.binlog(BOSSModel.this.symbols));
            }
          }
        }
      }
    });

    return words;
  }

  /**
   * Create the BOSS model for a fixed window-length and SFA word length
   *
   * @param words      the SFA words of the time series
   * @param samples    the samples to be transformed
   * @param wordLength the SFA word length
   * @return returns a BOSS model for each time series in samples
   */
  public BagOfPattern[] createBagOfPattern(
      final int[][] words,
      final TimeSeries[] samples,
      final int wordLength) {
    BagOfPattern[] bagOfPatterns = new BagOfPattern[words.length];

    final byte usedBits = (byte) Words.binlog(this.symbols);
    // FIXME
    // final long mask = (usedBits << wordLength) - 1l;
    final long mask = (1L << (usedBits * wordLength)) - 1L;

    // iterate all samples
    for (int j = 0; j < words.length; j++) {
      bagOfPatterns[j] = new BagOfPattern(words[j].length, samples[j].getLabel());

      // create subsequences
      long lastWord = Long.MIN_VALUE;

      for (int offset = 0; offset < words[j].length; offset++) {
        // use the words of larger length to get words of smaller lengths
        long word = words[j][offset] & mask;
        if (word != lastWord) { // ignore adjacent samples
          bagOfPatterns[j].bag.putOrAdd((int) word, (short) 1, (short) 1);
        }
        lastWord = word;
      }
    }

    return bagOfPatterns;
  }
}
