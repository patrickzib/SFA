// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa;

import java.io.IOException;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;
import sfa.transformation.SFA;
import sfa.transformation.SFA.HistogramType;
import sfa.transformation.SFADistance;

/**
 * Performs a 1-NN search using the SFA lower bounding distance
 *
 */
@RunWith(JUnit4.class)
public class SFAMinDistanceTest {

  @Test
  public void testSFAMinDistance() throws IOException {

    int symbols = 8;
    int wordLength = 16;
    final boolean normMean = true;

    SFA sfa = new SFA(HistogramType.EQUI_DEPTH);
    SFADistance sfaDistance = new SFADistance(sfa);

    // Load the train/test splits
    ClassLoader classLoader = SFAWordsTest.class.getClassLoader();
    TimeSeries[] train = TimeSeriesLoader.loadDataset(classLoader.getResource("datasets/univariate/CBF/CBF_TRAIN").getFile());
    TimeSeries[] test = TimeSeriesLoader.loadDataset(classLoader.getResource("datasets/univariate/CBF/CBF_TEST").getFile());

    // train SFA representation
    short[][] wordsTrain = sfa.fitTransform(train, wordLength, symbols, normMean);

    double minDistance = Double.MAX_VALUE;
    double accuracy = 0.0;
    int best = 0;

    // all queries
    for (int q = 0; q < test.length; q++) {
      TimeSeries query = test[q];
      // approximation
      double[] dftQuery = sfa.transformation.transform(query, wordLength);

      // quantization
      short[] wordQuery = sfa.quantization(dftQuery);

      // perform 1-NN search using the lower bounding distance
      for (int t = 0; t < train.length; t++) {
        double distance = sfaDistance.getDistance(wordsTrain[t], wordQuery, dftQuery, normMean, minDistance);

        // check the real distance, if lower bounding distance is smaller than best-so-far
        if (distance < minDistance) {
          double realDistance = getEuclideanDistance(train[t], query, minDistance);
          if (realDistance < minDistance) {
            minDistance = realDistance;
            best = t;
          }

          // plausibility check
          Assert.assertTrue("Lower bounding violated:\tSFA: " + distance + "\tED: " + realDistance, realDistance > distance);
        }
      }

      if (test[q].getLabel().equals(train[best].getLabel())) {
        accuracy++;
      }
    }

    System.out.println("Accuracy: "+ String.format("%.3f", accuracy / test.length));
  }

  private static double getEuclideanDistance (TimeSeries t1, TimeSeries t2, double minValue) {
    double distance = 0;
    double[] t1Values = t1.getData();
    double[] t2Values = t2.getData();

    for (int i = 0; i < Math.min(t1.getLength(), t2.getLength()); i++) {
      double value = t1Values[i] - t2Values[i];
      distance += value*value;

      // pruning
      if (distance > minValue) {
        return Double.POSITIVE_INFINITY;
      }
    }

    return distance;
  }
}
