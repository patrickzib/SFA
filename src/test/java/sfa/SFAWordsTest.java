// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa;

import java.io.IOException;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;
import sfa.transformation.SFA;
import sfa.transformation.SFA.HistogramType;

/**
 * Performs a 1-NN search
 *
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
    TimeSeries[] train = TimeSeriesLoader
        .loadDataset(classLoader.getResource("datasets/CBF/CBF_TRAIN").getFile());
    TimeSeries[] test = TimeSeriesLoader
        .loadDataset(classLoader.getResource("datasets/CBF/CBF_TEST").getFile());

    // train SFA representation
    sfa.fitTransform(train, wordLength, symbols, normMean);

    // bins
    sfa.printBins();

    // transform
    for (int q = 0; q < test.length; q++) {
      short[] wordQuery = sfa.transform(test[q]);
      System.out.println("Time Series " + q + "\t" + toSfaWord(wordQuery));
    }
  }

  public static String toSfaWord(short[] word) {
    StringBuilder sfaWord = new StringBuilder();

    for (short c : word) {
      sfaWord.append((char)(Character.valueOf('a') + c));
    }

    return sfaWord.toString();
  }
}
