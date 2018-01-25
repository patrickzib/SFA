// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.transformation;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;

import java.io.IOException;
import java.util.Random;

/**
 * Tests the Momentary Fourier transformation
 */
@RunWith(JUnit4.class)
public class MFTTest {

  /**
   * Tests the Fourier transform of a time series
   *
   * @throws IOException
   */
  @Test
  public void testTransform() throws IOException {
    for (int windowSize : new int[]{4,16,19,32,33,64}) {
      TimeSeries query = TimeSeriesLoader.generateRandomWalkData(windowSize, new Random(1));

      for (int l : new int[]{2, 4, 5, 6, 8, 10, 12, 14, 16}) {
        for (boolean normMean : new boolean[]{true, false}) {
          for (boolean lowerBounding : new boolean[]{true, false}) {
            // generate a random sample
            TimeSeries ts = TimeSeriesLoader.generateRandomWalkData(windowSize, new Random(2));

            // transform using the Fourier Transform
            MFT mft = new MFT(windowSize, normMean, lowerBounding);
            double[] dftData = mft.transform(ts, l);

            Assert.assertEquals("Not enough Fourier coefficients", dftData.length, l);

            if (l > windowSize) {
              // if more Fourier coefficients are accessed than available,
              // the last ones should be 0
              for (int i = l - mft.getStartOffset(); i < windowSize; i++) {
                Assert.assertEquals("Non zero coefficients", dftData[i],0.0, 0);
              }
            }

            // if set, the DFT should lower bound the Euclidean distance on the real time series
            if (lowerBounding) {
              double[] queryDFTData = mft.transform(query, l);

              // ED
              double distance = 0.0;
              for (int i = 0; i < query.getLength(); i++) {
                distance += (query.getData(i) - ts.getData(i)) * (query.getData(i) - ts.getData(i));
              }

              // lower bounding distance
              double distance2 = 0.0;
              for (int i = 0; i < dftData.length; i++) {
                distance2 += (dftData[i] - queryDFTData[i]) * (dftData[i] - queryDFTData[i]);
              }

              // System.out.println(distance + " " + distance2);
              Assert.assertTrue("DFT distance does not lower bound true distance", distance > distance2 );
            }
          }
        }
      }
    }

    System.out.println("DFT tests done");
  }


  /**
   * Test for different parameter settings, if the O(n) Momentary Fourier Tranfsorm (MFT)
   * returns identical Fourier coefficients as the O(n log n) Discrete Fourier Transform (DFT)
   *
   * @throws IOException
   */
  @Test
  public void testTransformWindowing() throws IOException {

    // generate a random sample
    TimeSeries timeSeries = TimeSeriesLoader.generateRandomWalkData(1024, new Random());

    // test even window sizes
    // test uneven window sizes
    for (int windowSize : new int[]{4,16,19,32,33,64}) {
      // test l > windowSize
      // test uneven l
      for (int l : new int[]{2, 4, 5, 6, 8, 10, 12, 14, 16}) {
        for (boolean lowerBounding : new boolean[]{true, false}) {
          for (boolean normMean : new boolean[]{true, false}) {
            MFT mft = new MFT(windowSize, normMean, lowerBounding);
            double[][] mftData = mft.transformWindowing(timeSeries, l);
            TimeSeries[] subsequences = timeSeries.getSubsequences(windowSize, normMean);

            Assert.assertEquals("Not enough MFT transformations", mftData.length, subsequences.length);

            for (int i = 0; i < mftData.length; i++) {
              double[] dftData = mft.transform(subsequences[i], l);
              Assert.assertArrayEquals("DFT not equal to MFT for l: " + l, mftData[i], dftData, 0.01);
            }
          }
        }
      }
    }

    System.out.println("MFT tests done");
  }

  /**
   * Tests the incremental mean stddev calculations for window sizes larger than the data size
   *
   * @throws IOException
   */
  @Test
  public void testIncrementalMeanStddev() throws IOException {

    // generate a random sample
    TimeSeries timeSeries = TimeSeriesLoader.generateRandomWalkData(32, new Random());

    // test for different window sizes
    // including windowSize > timeSeries.length
    for (int windowSize : new int[]{16, 32, 64, 128}) {
      double[] means = new double[Math.max(timeSeries.getLength()-windowSize+1,1)];
      double[] stds = new double[Math.max(timeSeries.getLength()-windowSize+1,1)];
      TimeSeries.calcIncrementalMeanStddev(windowSize, timeSeries.getData(), means, stds);
      TimeSeries[] subsequences = timeSeries.getSubsequences(windowSize, false);

      Assert.assertEquals("Not enough mean values", means.length, subsequences.length);
      Assert.assertEquals("Not enough stddev values", stds.length, subsequences.length);

      for (int i = 0; i < subsequences.length; i++) {
        Assert.assertEquals("Mean not equal: ", means[i], subsequences[i].getMean(), 0.001);
        Assert.assertEquals("Stddev not equal: ", stds[i], subsequences[i].getStddev(), 0.001);
      }

    }

    System.out.println("Incremental Mean Stddev tests done");
  }

  /**
   * Test for different parameter settings, if the O(n) Momentary Fourier Tranfsorm (MFT)
   * returns identical Fourier coefficients as the O(n log n) Discrete Fourier Transform (DFT)
   *
   * @throws IOException
   */
  @Test
  public void testTransformLargerWindowSizes() throws IOException {

    // generate a random sample
    TimeSeries timeSeries = TimeSeriesLoader.generateRandomWalkData(32, new Random());
    TimeSeries.APPLY_Z_NORM = true;

    // test even window sizes
    // test uneven window sizes
    for (int windowSize : new int[]{64}) {
      // test l > windowSize
      // test uneven l
      for (int l : new int[]{10, 12, 14, 16}) {
        for (boolean lowerBounding : new boolean[]{true, false}) {
          for (boolean normMean : new boolean[]{true, false}) {
            MFT mft = new MFT(windowSize, normMean, lowerBounding);
            double[][] mftData = mft.transformWindowing(timeSeries, l);
            TimeSeries[] subsequences = timeSeries.getSubsequences(windowSize, normMean);

            Assert.assertEquals("Not enough MFT transformations", mftData.length, subsequences.length);

            for (int i = 0; i < mftData.length; i++) {
              double[] dftData = mft.transform(subsequences[i], l);
              Assert.assertArrayEquals("DFT not equal to MFT for l: " + l, mftData[i], dftData, 0.01);
            }
          }
        }
      }
    }

    System.out.println("Larger window sizes done");
  }
}
