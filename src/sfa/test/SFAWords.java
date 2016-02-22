// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.test;

import java.io.File;
import java.io.IOException;

import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;
import sfa.transformation.SFA;
import sfa.transformation.SFA.HistogramType;

/**
 * Performs a 1-NN search
 *
 */
public class SFAWords {
  
  public static void main(String[] argv) throws IOException {
    
    int symbols = 8;
    int wordLength = 16;
    boolean normMean = true;
    
    SFA sfa = new SFA(HistogramType.EQUI_DEPTH, normMean);    

    TimeSeries[] train = TimeSeriesLoader.loadDatset(new File("./datasets/CBF/CBF_TRAIN"));
    TimeSeries[] test = TimeSeriesLoader.loadDatset(new File("./datasets/CBF/CBF_TEST"));
    
    // train SFA representation
    sfa.fitTransform(train, wordLength, symbols);
   
    // bins
    sfa.printBins();
    
    // transform
    for (int q = 0; q < test.length; q++) {
      short[] wordQuery = sfa.transform(test[q]);      
      System.out.println("Time Series " + q + "\t" + toSfaWord(wordQuery));      
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
