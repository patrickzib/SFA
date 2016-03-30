// Copyright (c) 2016 - Patrick Schäfer (patrick.schaefer@zib.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa.test;

import java.io.File;
import java.io.IOException;

import sfa.classification.BOSSEnsembleClassifier;
import sfa.classification.BOSSVSClassifier;
import sfa.classification.Classifier;
import sfa.classification.ParallelFor;
import sfa.classification.ShotgunClassifier;
import sfa.classification.ShotgunEnsembleClassifier;
import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;

public class UCRClassification {

  // The datasets to use
  public static String[] datasets = new String[]{
    "Coffee",
    "CBF",
    "Beef",
  };

  public static void main(String argv[]) throws IOException {
    try {
      // the relative path to the datasets
      File dir = new File("./datasets/");

      for (String s : datasets) {
        File d = new File(dir.getAbsolutePath()+"/"+s);
        if (d.exists() && d.isDirectory()) {
          for (File f : d.listFiles()) {
            if (f.getName().toUpperCase().endsWith("TRAIN")) {
              File train = f;
              File test = new File(f.getAbsolutePath().replaceFirst("TRAIN", "TEST"));

              if (!test.exists()) {
                System.err.println("File " + test.getName() + " does not exist");
                test = null;
              }

              Classifier.DEBUG = false;
                  
              // Load the train/test splits
              TimeSeries[] testSamples = TimeSeriesLoader.loadDatset(test);
              TimeSeries[] trainSamples = TimeSeriesLoader.loadDatset(train);

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
              
//              // The Shotgun classifier
//              Classifier shotgun = new ShotgunClassifier(trainSamples, testSamples);
//              Classifier.Score scoreShotgun = shotgun.eval();
//              System.out.println(s + ";" + scoreShotgun.toString());
            }
          }
        }
        else {
          System.err.println("Does not exist!" + d.getAbsolutePath());
        }
      }
    } finally {
      ParallelFor.shutdown();
    }
  }

}