// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa;

import java.io.File;
import java.io.IOException;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import sfa.classification.BOSSEnsembleClassifier;
import sfa.classification.BOSSVSClassifier;
import sfa.classification.Classifier;
import sfa.classification.ParallelFor;
import sfa.classification.ShotgunClassifier;
import sfa.classification.ShotgunEnsembleClassifier;
import sfa.classification.WEASELClassifier;
import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;

@RunWith(JUnit4.class)
public class UCRClassificationTest {

  // The datasets to use
  public static String[] datasets = new String[] {
    "Coffee", "ECG200", "FaceFour", "OliveOil",
    "Gun_Point", "Beef",
    "DiatomSizeReduction",
    "CBF",
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

      //File dir = new File(classLoader.getResource("datasets/").getFile());
      File dir = new File("/Users/bzcschae/workspace/similarity/datasets/classification");

      for (String s : datasets) {
        File d = new File(dir.getAbsolutePath()+"/"+s);
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

              // The WEASEL-classifier
              Classifier w = new WEASELClassifier(trainSamples, testSamples);
              Classifier.Score scoreW = w.eval();
              System.out.println(s + ";" + scoreW.toString());

              // The BOSS ensemble classifier
              Classifier boss = new BOSSEnsembleClassifier(trainSamples, testSamples);
              Classifier.Score scoreBOSS = boss.eval();
              System.out.println(s + ";" + scoreBOSS.toString());

              // The BOSS VS classifier
              Classifier bossVS = new BOSSVSClassifier(trainSamples, testSamples);
              Classifier.Score scoreBOSSVS = bossVS.eval();
              System.out.println(s + ";" + scoreBOSSVS.toString());

              // The Shotgun ensemble classifier
              Classifier shotgunEnsemble = new ShotgunEnsembleClassifier(trainSamples, testSamples);
              Classifier.Score scoreShotgunEnsemble = shotgunEnsemble.eval();
              System.out.println(s + ";" + scoreShotgunEnsemble.toString());

              // The Shotgun classifier
              Classifier shotgun = new ShotgunClassifier(trainSamples, testSamples);
              Classifier.Score scoreShotgun = shotgun.eval();
              System.out.println(s + ";" + scoreShotgun.toString());
            }
          }
        }
        else {
          // not really an error. just a hint:
          System.err.println("Dataset could not be found: " + d.getAbsolutePath() + ". " +
              "Please download datasets from [http://www.cs.ucr.edu/~eamonn/time_series_data/].");
        }
      }
    } finally {
      ParallelFor.shutdown();
    }
  }
}
