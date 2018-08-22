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

/**
 * Tests the SFA transformation
 */
@RunWith(JUnit4.class)
public class SFAWordsTest {

  @Test
  public void testSFAWords() throws IOException {
    int symbols = 8;
    int wordLength = 16;
    boolean normMean = true;

    SFA sfa = new SFA(HistogramType.EQUI_DEPTH);

    // Load the train/test splits
    ClassLoader classLoader = SFAWordsTest.class.getClassLoader();
    TimeSeries[] train = TimeSeriesLoader.loadDataset(classLoader.getResource("datasets/univariate/CBF/CBF_TRAIN").getFile());
    TimeSeries[] test = TimeSeriesLoader.loadDataset(classLoader.getResource("datasets/univariate/CBF/CBF_TEST").getFile());

    // train SFA representation
    sfa.fitTransform(train, wordLength, symbols, normMean);

    // outout discretization bins
//    sfa.printBins();

    // check discretization bins
    Assert.assertTrue("Wrong word queryLength of SFA transformation detected", sfa.bins.length == wordLength);
    for (int i = 0; i < sfa.bins.length; i++) {
      for (int j = 0; j < sfa.bins[i].length-1; j++) {
        Assert.assertTrue("SFA bins should be monotonically increasing.", sfa.bins[i][j] <= sfa.bins[i][j+1]);
      }
    }

    // transform
    for (int q = 0; q < test.length; q++) {
      short[] wordQuery = sfa.transform(test[q]);
      Assert.assertTrue("SFA word queryLength does not match actual queryLength.", wordQuery.length == wordLength);

//      System.out.println(q + "-th transformed time series SFA word " + "\t" + toSfaWord(wordQuery, symbols));
    }
  }

  public static String toSfaWord(short[] word, int symbols) {
    StringBuilder sfaWord = new StringBuilder();

    for (short c : word) {
      sfaWord.append((char) (Character.valueOf('a') + c));
      Assert.assertTrue("Wrong symbols used ", c < symbols && c >= 0);
    }

    sfaWord.append("\t... OK");
    return sfaWord.toString();
  }
}
