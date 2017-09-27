// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;
import sfa.transformation.MFT;
import sfa.transformation.SFA;
import sfa.transformation.SFA.HistogramType;

import java.io.IOException;
import java.util.Random;

/**
 * Tests the Momentary Fourier transformation
 */
@RunWith(JUnit4.class)
public class MFTTest {

  @Test
  public void testMFT() throws IOException {
    int windowSize = 16;

    // generate a random sample
    TimeSeries timeSeries = TimeSeriesLoader.generateRandomWalkData(1024, new Random());

    for (int l : new int[]{16}) {
      for (boolean lowerBounding : new boolean[]{true}) {
        for (boolean normMean : new boolean[]{true}) {

          MFT mft = new MFT(windowSize, normMean, lowerBounding);
          double[][] mftData = mft.transformWindowing(timeSeries, l);
          TimeSeries[] subsequences = timeSeries.getSubsequences(windowSize, normMean);

          Assert.assertEquals("Not enough MFT transformations", mftData.length, subsequences.length);

          for (int i = 0; i < mftData.length; i++) {
            double[] dftData = mft.transform(subsequences[i], l);
            Assert.assertArrayEquals("DFT not equal to MFT for l: " + l, mftData[i], dftData, 0.04);
          }
        }
      }
    }

    System.out.println("MFT tests done");
  }
}
