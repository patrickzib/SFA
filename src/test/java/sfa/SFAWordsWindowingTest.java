// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa;

import java.io.File;
import java.io.IOException;

import org.junit.Test;
import org.junit.runners.JUnit4;
import org.junit.runner.RunWith;

import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;
import sfa.transformation.SFA;
import sfa.transformation.SFA.HistogramType;

/**
 * Performs a 1-NN search
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
        TimeSeries[] train = TimeSeriesLoader.loadDatset(
                new File(classLoader.getResource("datasets/CBF/CBF_TRAIN").getFile()));
        TimeSeries[] test = TimeSeriesLoader.loadDatset(
                new File(classLoader.getResource("datasets/CBF/CBF_TEST").getFile()));

        // train SFA representation
        sfa.fitWindowing(train, windowLength, wordLength, symbols, normMean, true);

        // bins
        sfa.printBins();

        // transform
        for (int q = 0; q < test.length; q++) {
            short[][] wordsQuery = sfa.transformWindowing(test[q], wordLength);
            System.out.print("Time Series " + q + "\t");

            for (short[] word : wordsQuery) {
                System.out.print(toSfaWord(word) + ";");
            }

            System.out.println("");
        }
    }

    public static String toSfaWord(short[] word) {
        StringBuffer sfaWord = new StringBuffer();

        for (short c : word) {
            sfaWord.append((char)(Character.valueOf('a') + c));
        }

        return sfaWord.toString();
    }
}
