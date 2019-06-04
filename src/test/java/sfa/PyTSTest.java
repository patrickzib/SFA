// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import sfa.classification.WEASELClassifier;
import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;
import sfa.transformation.MFT;
import sfa.transformation.SFA;
import sfa.transformation.SFASupervised;
import sfa.transformation.WEASEL;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

/**
 * Tests the Momentary Fourier transformation
 */
@RunWith(JUnit4.class)
public class PyTSTest {

  @Test
  public void testDftPyTS() throws IOException {
    boolean lowerBounding = false;
    boolean normMean = false;
    int l = 8;

    for (int windowSize : new int[]{/*4,*/16/*,19,32,33,64*/}) {
      TimeSeries ts = TimeSeriesLoader.generateRandomWalkData(windowSize, new Random(1));
      ts.norm(normMean);
      // transform using the Fourier Transform
      MFT mft = new MFT(windowSize, normMean, lowerBounding);
      double[] dftData = mft.transform(ts, l);

      System.out.println (Arrays.toString(dftData));
    }

    int symbols = 4;
    int wordLength = 4;

    SFA sfa = new SFA(SFA.HistogramType.EQUI_FREQUENCY);
    sfa.lowerBounding = false;

    // Load the train/test splits
    ClassLoader classLoader = SFAWordsTest.class.getClassLoader();
    TimeSeries[] train = TimeSeriesLoader.loadDataset(classLoader.getResource("datasets/univariate/CBF/CBF_TRAIN").getFile());

    //for (TimeSeries t : train) {
    //  System.out.println(Arrays.toString(t.getData())+",");
    //}

//    for (TimeSeries t : train) {
//      System.out.print(t.getLabel()+",");
//    }

    // transform using the Fourier Transform
    MFT mft = new MFT(10, normMean, lowerBounding);
    double[][] dftData = mft.transformWindowing(train[0], 5);
    for (double[] dft : dftData)
      System.out.println(Arrays.toString(dft));


//    int i = 0;
//    //for (TimeSeries ts : train) {
//    //  System.out.println((i++)+ " " + Arrays.toString(mft.transform(ts, wordLength)));
//    //}
//
//    // train SFA representation
//    short[][] words = sfa.fitTransform(train, wordLength, symbols, normMean);
//
//    // outout discretization bins
//    sfa.printBins();
//
//    for (short[] word : words) {
//      System.out.println(Arrays.toString(word));
//    }
//
//    SFASupervised sfa2 = new SFASupervised(SFA.HistogramType.EQUI_FREQUENCY);
//
//    // train SFA representation
//    short[][] words2 = sfa2.fitTransform(train, wordLength, symbols, normMean);
//
//    // outout discretization bins
//    sfa2.printBins();
//
//    for (short[] word : words2) {
//      System.out.println(Arrays.toString(word));
//    }
//
//    WEASEL weasel = new WEASEL(8, 8, new int[]{10, 16}, true, false);

  }

}
