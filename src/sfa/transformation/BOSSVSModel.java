// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.transformation;

import java.util.HashSet;

import com.carrotsearch.hppc.IntFloatOpenHashMap;
import com.carrotsearch.hppc.IntShortOpenHashMap;
import com.carrotsearch.hppc.ObjectObjectOpenHashMap;
import com.carrotsearch.hppc.cursors.FloatCursor;
import com.carrotsearch.hppc.cursors.IntFloatCursor;
import com.carrotsearch.hppc.cursors.IntIntCursor;
import com.carrotsearch.hppc.cursors.ObjectCursor;
import com.carrotsearch.hppc.cursors.ObjectObjectCursor;

/**
 * The Bag-of-SFA-Symbols in Vector Space model as published in
 *    Schäfer, P.: Scalable time series classification. DMKD (Preprint)
 *   
 * @author bzcschae
 *
 */
public class BOSSVSModel extends BOSSModel {
  
  public BOSSVSModel(int maxF, int maxS, int windowLength, boolean normMean) {
    super(maxF, maxS, windowLength, normMean);
  }
  
  public ObjectObjectOpenHashMap<String, IntFloatOpenHashMap> createTfIdf(
      final BagOfPattern[] bagOfPatterns,
      final int[] sampleIndices,
      final HashSet<String> uniqueLabels) {

    ObjectObjectOpenHashMap<String, IntFloatOpenHashMap> matrix = new ObjectObjectOpenHashMap<String, IntFloatOpenHashMap>(uniqueLabels.size());
    initMatrix(matrix, uniqueLabels, bagOfPatterns);

    for (int j : sampleIndices) {
      String label = bagOfPatterns[j] .label;
      IntFloatOpenHashMap wordInBagFreq = matrix.get(label);
      for (IntIntCursor key : bagOfPatterns[j].bag) {
        wordInBagFreq.putOrAdd(key.key, key.value, key.value);
      }
    }

    // count the number of classes where the word is present
    IntShortOpenHashMap wordInClassFreq = new IntShortOpenHashMap(matrix.iterator().next().value.size());

    for (ObjectCursor<IntFloatOpenHashMap> stat : matrix.values()) {
      // count the occurence of words
      for (IntFloatCursor key : stat.value) {
        wordInClassFreq.putOrAdd(key.key, (short)1, (short)1);
      }
    }

    // calculate the tfIDF value for each class
    for (ObjectObjectCursor<String, IntFloatOpenHashMap> stat : matrix) {
      IntFloatOpenHashMap tfIDFs = stat.value;
      // calculate the tfIDF value for each word
      for (IntFloatCursor patternFrequency : tfIDFs) {
        short wordCount = wordInClassFreq.get(patternFrequency.key);
        if (patternFrequency.value > 0
            && uniqueLabels.size() != wordCount // avoid Math.log(1)
            ) {
          double tfValue = 1.0+Math.log10(patternFrequency.value); // smoothing
          double idfValue = Math.log10(1.0+uniqueLabels.size() / (double)wordCount); // smoothing
          double tfIdf = tfValue / idfValue;

          // update the tfIDF vector
          tfIDFs.values[patternFrequency.index] = (float) tfIdf;
        }
        else {
          tfIDFs.values[patternFrequency.index] = 0;
        }
      }
    }
    
    normalizeTfIdf(matrix);
    
    return matrix;
  }
  
  protected void initMatrix(
      final ObjectObjectOpenHashMap<String, IntFloatOpenHashMap> matrix,
      final HashSet<String> uniqueLabels, 
      final BagOfPattern[] bag) {
    for (String label : uniqueLabels) {
      IntFloatOpenHashMap stat = matrix.get(label);
      if (stat == null) {
        matrix.put(label, new IntFloatOpenHashMap(bag[0].bag.size()*bag.length));
      } else {
        if (stat != null) {
          stat.clear();
        }
      }
    }
  }
  
  public void normalizeTfIdf(final ObjectObjectOpenHashMap<String, IntFloatOpenHashMap> classStatistics) {
    for (ObjectCursor<IntFloatOpenHashMap> classStat : classStatistics.values()) {
      double squareSum = 0.0;
      for (FloatCursor entry : classStat.value.values()) {
        squareSum += entry.value*entry.value;
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
