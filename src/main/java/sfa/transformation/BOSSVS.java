// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.transformation;

import com.carrotsearch.hppc.IntFloatHashMap;
import com.carrotsearch.hppc.IntShortHashMap;
import com.carrotsearch.hppc.ObjectObjectHashMap;
import com.carrotsearch.hppc.cursors.*;

import java.util.Set;

/**
 * The Bag-of-SFA-Symbols in Vector Space boss as published in
 * Schäfer, P.: Scalable time series classification. DMKD (Preprint)
 *
 */
public class BOSSVS extends BOSS {

  public BOSSVS(){}

  /**
   * Create a BOSS VS boss.
   *
   * @param maxF         queryLength of the SFA words
   * @param maxS         alphabet size
   * @param windowLength subsequence (window) queryLength used for extracting SFA words from
   *                     time series.
   * @param normMean     set to true, if mean should be set to 0 for a window
   */
  public BOSSVS(int maxF, int maxS, int windowLength, boolean normMean) {
    super(maxF, maxS, windowLength, normMean);
  }

  public ObjectObjectHashMap<Double, IntFloatHashMap> createTfIdf(
      final BagOfPattern[] bagOfPatterns,
      final Set<Double> uniqueLabels) {
    int[] sampleIndices = createIndices(bagOfPatterns.length);
    return createTfIdf(bagOfPatterns, sampleIndices, uniqueLabels);
  }

  protected static int[] createIndices(int length) {
    int[] indices = new int[length];
    for (int i = 0; i < length; i++) {
      indices[i] = i;
    }
    return indices;
  }

  /**
   * Obtains the TF-IDF representation based on the BOSS representation. Only those elements
   * in sampleIndices are used (useful for cross-validation).
   *
   * @param bagOfPatterns The BOSS (bag-of-patterns) representation of the time series
   * @param sampleIndices The indices to use
   * @param uniqueLabels  The unique class labels in the data set
   * @return              returns the tf-idf boss for the time series
   */
  public ObjectObjectHashMap<Double, IntFloatHashMap> createTfIdf(
      final BagOfPattern[] bagOfPatterns,
      final int[] sampleIndices,
      final Set<Double> uniqueLabels) {

    ObjectObjectHashMap<Double, IntFloatHashMap> matrix = new ObjectObjectHashMap<>(
        uniqueLabels.size());
    initMatrix(matrix, uniqueLabels, bagOfPatterns);

    for (int j : sampleIndices) {
      Double label = bagOfPatterns[j].label;
      IntFloatHashMap wordInBagFreq = matrix.get(label);
      for (IntIntCursor key : bagOfPatterns[j].bag) {
        wordInBagFreq.putOrAdd(key.key, key.value, key.value);
      }
    }

    // count the number of classes where the word is present
    IntShortHashMap wordInClassFreq = new IntShortHashMap(matrix.iterator().next().value.size());

    for (ObjectCursor<IntFloatHashMap> stat : matrix.values()) {
      // count the occurrence of words
      for (IntFloatCursor key : stat.value) {
        wordInClassFreq.putOrAdd(key.key, (short) 1, (short) 1);
      }
    }

    // calculate the tfIDF value for each class
    for (ObjectObjectCursor<Double, IntFloatHashMap> stat : matrix) {
      IntFloatHashMap tfIDFs = stat.value;
      // calculate the tfIDF value for each word
      for (IntFloatCursor patternFrequency : tfIDFs) {
        short wordCount = wordInClassFreq.get(patternFrequency.key);
        if (patternFrequency.value > 0
            && uniqueLabels.size() != wordCount // avoid Math.log(1)
            ) {
          double tfValue = 1.0 + Math.log10(patternFrequency.value); // smoothing
          double idfValue = Math.log10(1.0 + uniqueLabels.size() / (double) wordCount); // smoothing
          double tfIdf = tfValue / idfValue;

          // update the tfIDF vector
          tfIDFs.values[patternFrequency.index] = (float) tfIdf;
        } else {
          tfIDFs.values[patternFrequency.index] = 0;
        }
      }
    }

    // norm the tf-idf-matrix
    normalizeTfIdf(matrix);

    return matrix;
  }

  protected void initMatrix(
      final ObjectObjectHashMap<Double, IntFloatHashMap> matrix,
      final Set<Double> uniqueLabels,
      final BagOfPattern[] bag) {
    for (Double label : uniqueLabels) {
      IntFloatHashMap stat = matrix.get(label);
      if (stat == null) {
        matrix.put(label, new IntFloatHashMap(bag[0].bag.size() * bag.length));
      } else {
        stat.clear();
      }
    }
  }

  /**
   * Norm the vector to queryLength 1
   *
   * @param classStatistics
   */
  public void normalizeTfIdf(final ObjectObjectHashMap<Double, IntFloatHashMap> classStatistics) {
    for (ObjectCursor<IntFloatHashMap> classStat : classStatistics.values()) {
      double squareSum = 0.0;
      for (FloatCursor entry : classStat.value.values()) {
        squareSum += entry.value * entry.value;
      }
      double squareRoot = Math.sqrt(squareSum);
      if (squareRoot > 0) {
        for (FloatCursor entry : classStat.value.values()) {
          //entry.value /= squareRoot;
          classStat.value.values[entry.index] /= squareRoot;
        }
      }
    }
  }
}
