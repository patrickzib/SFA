// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa;

import org.junit.Assert;
import org.junit.Test;
import org.junit.internal.ArrayComparisonFailure;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;
import sfa.transformation.SFA;
import sfa.transformation.SFA.HistogramType;
import sfa.transformation.SFASupervised;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;

/**
 * Extracts windows from a time series and transforms each window using SFA.
 *
 */
@RunWith(JUnit4.class)
public class SFASupervisedReproducibleIntervals {

  @Test
  public void testSFAWordsWindowing() {
    int symbols = 4;
    int wordLength = 4;
    boolean normMean = true;


    // Load the train/test splits
    ClassLoader classLoader = SFAWordsTest.class.getClassLoader();
    TimeSeries[] train = TimeSeriesLoader.loadDataset(classLoader.getResource("datasets/univariate/BEEF/BEEF_TRAIN").getFile());

    // test 120 window sizes
    for (int w = 4; w < 120; w++) {
      double[][] bins = null;
      int[] bestValues = null;
      ArrayList<SFA.ValueLabel>[] orderLine = null;

      // test for reproducible splits
      for (int i = 0; i < 3; i++) {
        SFASupervised sfa = new SFASupervised(HistogramType.INFORMATION_GAIN);

        // train SFA representation
        sfa.fitWindowing(train, w, wordLength, symbols, normMean, false);

        // bins
        if (bins != null) {
          try {
            Assert.assertArrayEquals(bins, sfa.bins);
          } catch (ArrayComparisonFailure ex) {

            System.out.println(Arrays.toString(sfa.bestValues));
            System.out.println(Arrays.toString(bestValues));


//            output :
//            for (int a = 0; a < bins.length; a++) {
//              for (int b = 0; b < bins[a].length; b++) {
//                if (bins[a][b]!=sfa.bins[a][b]) {
//                  System.out.println(orderLine[a]);
//                  System.out.println(sfa.orderLine[a]);
//
//                  ArrayList<Integer> splitPoints = new ArrayList<>();
//                  sfa.findBestSplit(orderLine[a], 0, orderLine[a].size(), symbols, splitPoints);
//                  System.out.println(Arrays.toString(splitPoints.toArray(new Integer[]{})));
//
//                  // test if the split points are the same
//                  for (int x = 0; x < 10; x++) {
//                    splitPoints = new ArrayList<>();
//                    sfa.findBestSplit(orderLine[a], 0, orderLine[a].size(), symbols, splitPoints);
//
//                    System.out.println("Splitpoints");
//                    for (int split : splitPoints) {
//                      System.out.println(orderLine[a].get(split + 1).value);
//                    }
//                  }
//
//                  break output;
//                }
//              }
//            }


            System.out.println(w);

            sfa.printBins();
            sfa.bins = bins;
            sfa.printBins();

            throw ex;
          }
        }

        bestValues = sfa.bestValues;
        bins = sfa.bins;
        orderLine = sfa.orderLine;
      }
    }

    System.out.println("Test passed");
  }


  public static String toSfaWord(short[] word, int symbols) {
    StringBuilder sfaWord = new StringBuilder();

    for (short c : word) {
      sfaWord.append((char) (Character.valueOf('a') + c));
      Assert.assertTrue("Wrong symbols used ", c < symbols && c >= 0);
    }

    return sfaWord.toString();
  }
}
