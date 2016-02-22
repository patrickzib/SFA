package sfa.test;

import java.io.File;
import java.io.IOException;

import sfa.classification.BOSSEnsembleClassifier;
import sfa.classification.BOSSVSClassifier;
import sfa.classification.Classifier;
import sfa.classification.ParallelFor;
import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;

public class UCRClassification {

  public static String[] datasets = new String[]{
    "Coffee",
    "CBF",
    "Beef",
  };

  public static void main(String argv[]) throws IOException {
    try {
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
                  
              TimeSeries[] testSamples = TimeSeriesLoader.loadDatset(test);
              TimeSeries[] trainSamples = TimeSeriesLoader.loadDatset(train);

              BOSSEnsembleClassifier boss = new BOSSEnsembleClassifier(trainSamples, testSamples);
              BOSSEnsembleClassifier.Score scoreBOSS = boss.eval();
              System.out.println(s + ";" + scoreBOSS.toString());
              
              BOSSVSClassifier bossVS = new BOSSVSClassifier(trainSamples, testSamples);
              BOSSVSClassifier.Score scoreBOSSVS = bossVS.eval();
              System.out.println(s + ";" + scoreBOSSVS.toString());
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