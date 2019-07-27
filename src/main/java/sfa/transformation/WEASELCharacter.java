// Copyright (c) 2017 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.transformation;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.commons.math3.distribution.FDistribution;

import com.carrotsearch.hppc.IntLongHashMap;
import com.carrotsearch.hppc.LongFloatHashMap;
import com.carrotsearch.hppc.LongHashSet;
import com.carrotsearch.hppc.LongIntHashMap;
import com.carrotsearch.hppc.LongLongHashMap;
import com.carrotsearch.hppc.LongObjectHashMap;
import com.carrotsearch.hppc.cursors.LongFloatCursor;
import com.carrotsearch.hppc.cursors.LongIntCursor;

import sfa.classification.Classifier.Words;
import sfa.classification.ParallelFor;
import sfa.classification.WEASELCharacterClassifier;
import sfa.timeseries.TimeSeries;
import subwordTransformer.Parameter;
import subwordTransformer.SubwordTransformer;

/**
 * The WEASEL-Model as published in
 * <p>
 * Schäfer, P., Leser, U.: Fast and Accurate Time Series Classification with
 * WEASEL. CIKM 2017
 */
public class WEASELCharacter {

  public int alphabetSize;
  public int maxF;

  public int[] windowLengths;
  public boolean normMean;
  public boolean lowerBounding;
  public SFA[] signature;
  public Dictionary dict;

  public SubwordTransformer[] transformers;
  public int outputAlphabetSize;

  public final static int BLOCKS;

  static {
    Runtime runtime = Runtime.getRuntime();
    if (runtime.availableProcessors() <= 4) {
      BLOCKS = 8;
    } else {
      BLOCKS = runtime.availableProcessors();
    }

    // BLOCKS = 1; // for testing purposes
  }

  public WEASELCharacter() {
  }

  /**
   * Create a WEASEL model.
   *
   * @param maxF          Length of the SFA words
   * @param maxS          alphabet size
   * @param windowLengths the set of window lengths to use for extracting SFA
   *                      words from time series.
   * @param normMean      set to true, if mean should be set to 0 for a window
   * @param lowerBounding set to true, if the Fourier transform should be normed
   *                      (typically used to lower bound / mimic Euclidean
   *                      distance).
   */
  private WEASELCharacter(int maxF, int maxS, int[] windowLengths, boolean normMean, boolean lowerBounding) {
    this.maxF = maxF;
    this.alphabetSize = maxS;
    this.windowLengths = windowLengths;
    this.normMean = normMean;
    this.lowerBounding = lowerBounding;
    this.dict = new Dictionary();
    this.signature = new SFA[windowLengths.length];
  }

  public WEASELCharacter(int maxF, int maxS, int[] windowLengths, boolean normMean, boolean lowerBounding, SubwordTransformer<? extends Parameter> transformer) {
    this(maxF, maxS, windowLengths, normMean, lowerBounding);
    this.transformers = new SubwordTransformer[windowLengths.length];
    for (int i = 0; i < windowLengths.length; i++) {
      this.transformers[i] = transformer.clone();
    }
    this.outputAlphabetSize = this.transformers[0].getOutputAlphabetSize();
  }

  public WEASELCharacter(int maxF, int maxS, int[] windowLengths, boolean normMean, boolean lowerBounding, int outputAlphabetSize) {
    this(maxF, maxS, windowLengths, normMean, lowerBounding);
    this.outputAlphabetSize = outputAlphabetSize;
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
  public short[/* windowLength */][/* sample */][/* offset */][/* character */] createWords(final TimeSeries[] samples) {
    // create bag of words for each window queryLength
    final short[][][][] words = new short[this.windowLengths.length][samples.length][][];
    ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        for (int w = 0; w < WEASELCharacter.this.windowLengths.length; w++) {
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
  public short[][][] createWords(final TimeSeries[] samples, final int index) {

    // SFA quantization
    if (this.signature[index] == null) {
      this.signature[index] = new SFASupervised();
      this.signature[index].fitWindowing(samples, this.windowLengths[index], this.maxF, this.alphabetSize, this.normMean, this.lowerBounding);
    }

    // create words
    final short[][][] words = new short[samples.length][][];
    for (int i = 0; i < samples.length; i++) {
      if (samples[i].getLength() >= this.windowLengths[index]) {
        words[i] = this.signature[index].transformWindowing(samples[i]);
      } else {
        words[i] = new short[][] {};
      }
    }

    return words;
  }

  public void setTransformerTrainingWords(short[][][][] words) {
    ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        for (int i = 0; i < words.length; i++) {
          if (i % BLOCKS == id) {
            List<short[]> wordsList = new ArrayList<>();
            for (int j = 0; j < words[i].length; j++) {
              for (int k = 0; k < words[i][j].length; k++) {
                wordsList.add(words[i][j][k]);
              }
            }
            short[][] wordsArray = new short[wordsList.size()][];
            wordsArray = wordsList.toArray(wordsArray);
            WEASELCharacter.this.transformers[i].setWords(wordsArray);
          }
        }
      }
    });
  }

  public void fitSubwords(Parameter param) {
    ParallelFor.withIndex(BLOCKS, new ParallelFor.Each() {
      @Override
      public void run(int id, AtomicInteger processed) {
        for (int i = 0; i < WEASELCharacter.this.transformers.length; i++) {
          if (i % BLOCKS == id) {
            WEASELCharacter.this.transformers[i].fitParameter(param);
          }
        }
      }
    });
  }

  public int[][][] transformSubwordsOneWindow(short[][][] words, int w) {
    byte neededBits = (byte) Words.binlogRoundedUp(this.outputAlphabetSize);
    int[][][] intWords = new int[words.length][][];

    for (int sample = 0; sample < words.length; sample++) {
      intWords[sample] = new int[words[sample].length][];
      for (int offset = 0; offset < words[sample].length; offset++) {
        short[] word = words[sample][offset];
        short[][] subwords = this.transformers[w].transformWord(word);
        // if (subwords.length == 0) { // TODO: always add word itself?
        // intWords[sample][offset] = new int[] { (int) Words.createWord(word,
        // word.length, neededBits) };
        // } else {
        intWords[sample][offset] = new int[subwords.length + 1];
        for (int subwordIndex = 0; subwordIndex < subwords.length; subwordIndex++) {
          intWords[sample][offset][subwordIndex] = (int) Words.createWord(subwords[subwordIndex], subwords[subwordIndex].length, neededBits);
        }
        intWords[sample][offset][subwords.length] = (int) Words.createWord(word, word.length, neededBits);
        // }
      }
    }

    return intWords;
  }

  /**
   * Create words and bi-grams for all window lengths
   */
  public BagOfBigrams[] createBagOfPatterns(final int[][][] wordsForWindowLength, final TimeSeries[] samples, final int w, // index of used windowSize
      final int wordLength) {
    BagOfBigrams[] bagOfPatterns = new BagOfBigrams[samples.length];

    final byte usedBits = (byte) Words.binlogRoundedUp(this.outputAlphabetSize);
    final long mask = (1L << (usedBits * wordLength)) - 1L;
    int highestBit = Words.binlog(Integer.highestOneBit(WEASELCharacterClassifier.MAX_WINDOW_LENGTH)) + 1;

    // iterate all samples
    // and create a bag of pattern
    for (int j = 0; j < samples.length; j++) {
      bagOfPatterns[j] = new BagOfBigrams(wordsForWindowLength[j].length * 2, samples[j].getLabel());

      // create subsequences
      for (int offset = 0; offset < wordsForWindowLength[j].length; offset++) {
        for (int subword = 0; subword < wordsForWindowLength[j][offset].length; subword++) {
          long word = (wordsForWindowLength[j][offset][subword] & mask) << highestBit | w;
          bagOfPatterns[j].bob.putOrAdd(word, 1, 1);

          // add 2 grams
          int prevOffset = offset - this.windowLengths[w];
          if (prevOffset >= 0) {
            for (int prevSubword = 0; prevSubword < wordsForWindowLength[j][prevOffset].length; prevSubword++) {
              long prevWord = (wordsForWindowLength[j][prevOffset][prevSubword] & mask);
              if (prevWord != 0) {
                long newWord = (prevWord << 32 | word);
                bagOfPatterns[j].bob.putOrAdd(newWord, 1, 1);
              }
            }
          }

        }
      }
    }

    return bagOfPatterns;
  }

  public void trainAnova(final BagOfBigrams[] bob, double p_value) {

    int highestIndex = 0;
    IntLongHashMap reverseMap = new IntLongHashMap();
    Map<Double, List<LongLongHashMap>> classes = new HashMap<>();
    for (int j = 0; j < bob.length; j++) {
      List<LongLongHashMap> allTs = classes.get(bob[j].label);
      if (allTs == null) {
        allTs = new ArrayList<>();
        classes.put(bob[j].label, allTs);
      }
      LongLongHashMap keys = new LongLongHashMap(bob[j].bob.size()); // ugly to copy everything ...
      for (LongIntCursor word : bob[j].bob) {
        int index = dict.getWordIndex(word.key);
        reverseMap.put(index, word.key);
        keys.put(index, word.value);
        highestIndex = Math.max(index, highestIndex);
      }
      allTs.add(keys);
    }

    // dense double array
    highestIndex = highestIndex + 1;

    double nSamples = bob.length;
    double nClasses = classes.keySet().size();

    double[] f = SFASupervised.getFonewaySparse(highestIndex, classes, nSamples, nClasses);

    final FDistribution fdist = new FDistribution(null, nClasses - 1, nSamples - nClasses);
    for (int i = 0; i < f.length; i++) {
      f[i] = 1.0 - fdist.cumulativeProbability(f[i]);
    }

    // sort by largest f-value
    @SuppressWarnings("unchecked")
    List<SFASupervised.Indices<Double>> best = new ArrayList<>(f.length);
    for (int i = 0; i < f.length; i++) {
      if (!Double.isNaN(f[i]) && f[i] > 0.5) {
        best.add(new SFASupervised.Indices<>(i, f[i]));
      }
    }

    // Collections.sort(best);
    // best = best.subList(0, (int) Math.min(100, best.size()));

    LongHashSet bestWords = new LongHashSet();
    for (SFASupervised.Indices<Double> index : best) {
      bestWords.add(reverseMap.get(index.value.intValue()));
    }

    for (int j = 0; j < bob.length; j++) {
      for (LongIntCursor cursor : bob[j].bob) {
        if (!bestWords.contains(cursor.key)) {
          bob[j].bob.values[cursor.index] = 0;
        }
      }

    }

  }

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

        if (newChi > 0 && newChi >= chi_limit && !chiSquare.contains(feature.key)) {
          chiSquare.add(feature.key);
          pvalues.add(new PValueKey(newChi, feature.key));
        }
      }
    }

//  // limit to 100 (?) features per window size
//  int limit = 100;
//  if (pvalues.size() > limit) {
//    // sort by chi-squared value
//    Collections.sort(pvalues, new Comparator<PValueKey>() {
//      @Override
//      public int compare(PValueKey o1, PValueKey o2) {
//        int comp = -Double.compare(o1.pvalue, o2.pvalue);
//        if (comp!=0) { // tie breaker
//          return comp;
//        }
//        return Long.compare(o1.key, o2.key);
//      }
//    });
//    // only keep the best features (with highest chi-squared pvalue)
//    LongHashSet chiSquaredBest = new LongHashSet();
//    int count = 0;
//    double lastValue = 0.0;
//    for (PValueKey key : pvalues) {
//      chiSquaredBest.add(key.key);
//      if (++count >= Math.min(pvalues.size(), limit)
//        // keep all keys with the same values to solve ties
//        && key.pvalue != lastValue) {
//        break;
//      }
//      lastValue = key.pvalue;
//    }
//    chiSquare = chiSquaredBest;
//  }

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