// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import sfa.classification.*;
import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;

import java.io.File;
import java.io.IOException;

@RunWith(JUnit4.class)
public class EarlyClassificationTest {

  // The datasets to use
  public static String[] datasets = new String[]{
          "Coffee",
          "CBF",
          "Beef",
          "ECG200", "FaceFour", "OliveOil",
          "Gun_Point",
          "DiatomSizeReduction",
          "ECGFiveDays",
          "TwoLeadECG",
          "SonyAIBORobot SurfaceII",
          "MoteStrain",
          "ItalyPowerDemand",
          "SonyAIBORobot Surface",
  };

  @Test
  public void testUCRClassification() throws IOException {
    try {
      // the relative path to the datasets
      ClassLoader classLoader = SFAWordsTest.class.getClassLoader();

      File dir = new File(classLoader.getResource("datasets/univariate/").getFile());
      //File dir = new File("/Users/bzcschae/workspace/similarity/datasets/classification");

      TimeSeries.APPLY_Z_NORM = false;

      for (String s : datasets) {
        File d = new File(dir.getAbsolutePath() + "/" + s);
        if (d.exists() && d.isDirectory()) {
          for (File train : d.listFiles()) {
            if (train.getName().toUpperCase().endsWith("TRAIN")) {
              File test = new File(train.getAbsolutePath().replaceFirst("TRAIN", "TEST"));

              if (!test.exists()) {
                System.err.println("File " + test.getName() + " does not exist");
                test = null;
              }

              Classifier.DEBUG = false;

              // Load the train/test splits
              TimeSeries[] testSamples = TimeSeriesLoader.loadDataset(test);
              TimeSeries[] trainSamples = TimeSeriesLoader.loadDataset(train);

              // The TEASER-classifier
              TEASERClassifier t = new TEASERClassifier();
              TEASERClassifier.S = 20.0;

              Classifier.Score scoreT = t.eval(trainSamples, testSamples);
              System.out.println(s + ";" + scoreT.toString());

            }
          }
        } else {
          // not really an error. just a hint:
          System.out.println("Dataset could not be found: " + d.getAbsolutePath() + ". " +
                  "Please download datasets from [http://www.cs.ucr.edu/~eamonn/time_series_data/].");
        }
      }
    } finally {
      TimeSeries.APPLY_Z_NORM = true; // FIXME static variable breaks some test cases!
    }
  }
}
