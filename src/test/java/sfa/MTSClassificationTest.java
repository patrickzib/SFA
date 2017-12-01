// Copyright (c) 2017 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import sfa.classification.*;
import sfa.timeseries.MultiVariateTimeSeries;
import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;

import java.io.File;
import java.io.IOException;

@RunWith(JUnit4.class)
public class MTSClassificationTest {

  // The datasets to use
  public static String[] datasets = new String[]{
      "LP1",
      "LP2",
      "LP3",
      "LP4",
      "LP5",
      "PenDigits",
      "ShapesRandom",
      "DigitShapeRandom",
      "CMUsubject16",
      "ECG",
      "JapaneseVowels",
      "KickvsPunch",
      "Libras",
      "UWave",
      "Wafer",
      "WalkvsRun",
      "CharacterTrajectories",
      "ArabicDigits",
      "AUSLAN",
      "NetFlow",
  };


  @Test
  public void testUCRClassification() throws IOException {
    try {
      // the relative path to the datasets
      ClassLoader classLoader = SFAWordsTest.class.getClassLoader();

      File dir = new File(classLoader.getResource("datasets/multivariate/").getFile());

      for (String s : datasets) {
        File d = new File(dir.getAbsolutePath() + "/" + s);
        if (d.exists() && d.isDirectory()) {
          for (File train : d.listFiles()) {
            if (train.getName().toUpperCase().endsWith("TRAIN3")) {
              File test = new File(train.getAbsolutePath().replaceFirst("TRAIN3", "TEST3"));

              if (!test.exists()) {
                System.err.println("File " + test.getName() + " does not exist");
                test = null;
              }

              Classifier.DEBUG = false;

              boolean useDeltas = true;
              MultiVariateTimeSeries[] trainSamples = TimeSeriesLoader.loadMultivariateDatset(train, useDeltas);
              MultiVariateTimeSeries[] testSamples = TimeSeriesLoader.loadMultivariateDatset(test, useDeltas);

              MUSEClassifier weasel = new MUSEClassifier();
              MUSEClassifier.Score weaselScore = weasel.eval(trainSamples, testSamples);
              System.out.println(s + ";" + weaselScore.toString());
            }
          }
        } else{
          // not really an error. just a hint:
          System.out.println("Dataset could not be found: " + d.getAbsolutePath() + ".");
        }
      }
    } finally {
      ParallelFor.shutdown();
    }
  }
}
