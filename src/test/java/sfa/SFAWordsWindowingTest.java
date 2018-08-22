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
 * Extracts windows from a time series and transforms each window using SFA.
 *
 */
@RunWith(JUnit4.class)
public class SFAWordsWindowingTest {

  @Test
  public void testSFAWordsWindowing() throws IOException {
    int symbols = 4;
    int wordLength = 4;
    int windowLength = 64;
    boolean normMean = true;

    SFA sfa = new SFA(HistogramType.EQUI_DEPTH);

    // Load the train/test splits
    ClassLoader classLoader = SFAWordsTest.class.getClassLoader();
    TimeSeries[] train = TimeSeriesLoader.loadDataset(classLoader.getResource("datasets/univariate/CBF/CBF_TRAIN").getFile());
    TimeSeries[] test = TimeSeriesLoader.loadDataset(classLoader.getResource("datasets/univariate/CBF/CBF_TEST").getFile());

    // train SFA representation
    sfa.fitWindowing(train, windowLength, wordLength, symbols, normMean, true);

    // bins
    //sfa.printBins();

    // transform
    for (int q = 0; q < test.length; q++) {
      short[][] wordsQuery = sfa.transformWindowing(test[q]);
      //System.out.print(q + "-th time Series " + "\t");
      Assert.assertTrue("SFA word queryLength does not match actual queryLength.",
          wordsQuery.length == test[q].getLength()-windowLength+1);

      for (short[] word : wordsQuery) {
        Assert.assertTrue("SFA word queryLength does not match actual queryLength.", word.length == wordLength);
        //System.out.print(toSfaWord(word, symbols) + ";");
      }

      //System.out.println("");
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
