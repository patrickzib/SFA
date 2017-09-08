// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import org.junit.Test;
import org.junit.runners.JUnit4;
import org.junit.runner.RunWith;
import org.junit.Assert;

import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;
import sfa.transformation.SFA;
import sfa.transformation.SFA.HistogramType;

/**
 * Performs a 1-NN search
 *
 */
@RunWith(JUnit4.class)
public class SFAWordsVariableLength {

    @Test
    public void testSFAWordsVarLength() throws IOException {

        int symbols = 8;
        int wordLength = 16;
        boolean normMean = true;

        SFA sfa = new SFA(HistogramType.EQUI_DEPTH);

        // Load the train/test splits
        ClassLoader classLoader = SFAWords.class.getClassLoader();
        TimeSeries[] train = TimeSeriesLoader
                             .loadDatset(new File(classLoader.getResource("datasets/CBF/CBF_TRAIN")
                                         .getFile()));
        TimeSeries[] test = TimeSeriesLoader
                            .loadDatset(new File(classLoader.getResource("datasets/CBF/CBF_TEST")
                                        .getFile()));

        // train SFA representation using wordLength
        sfa.fitTransform(train, wordLength, symbols, normMean);

        // bins
        sfa.printBins();

        // transform
        for (int q = 0; q < test.length; q++) {
            short[] wordQuery = sfa.transform(test[q]);

            // iterate variable lengths
            for (int length = 4; length <= wordLength; length*=2) {
                System.out.println("Time Series " + q + "\t" + length + "\t" + toSfaWord(Arrays.copyOf(wordQuery, length)));
            }
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
