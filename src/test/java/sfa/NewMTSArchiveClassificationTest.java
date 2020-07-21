// Copyright (c) 2017 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)
package sfa;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import sfa.classification.Classifier;
import sfa.classification.MUSEClassifier;
import sfa.timeseries.MultiVariateTimeSeries;
import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.TreeMap;

@RunWith(JUnit4.class)
public class NewMTSArchiveClassificationTest {

  // The new multivariate datasets to use from
  // http://www.timeseriesclassification.com/dataset.php
  public static String[] datasets = new String[]{
      "ArticularyWordRecognition",
      "AtrialFibrillation",
      "BasicMotions",
      "CharacterTrajectories",
      "Cricket",
      "ERing",
      "Epilepsy",
      "EthanolConcentration",
      "FingerMovements",
      "HandMovementDirection",
      "Handwriting",
      "Heartbeat",
      "JapaneseVowels",
      "LSST",
      "Libras",
      "NATOPS",
      "PenDigits",
      "PhonemeSpectra",
      "RacketSports",
      "SelfRegulationSCP1",
      "SelfRegulationSCP2",
      "SpokenArabicDigits",
      "StandWalkJump",
      "UWaveGestureLibrary"

      //  Large "EigenWorms",
      //  Large "DuckDuckGeese",
      //  Large "PEMS-SF",
      //  Large "MotorImagery",
      //  Large "InsectWingbeat",
      //  Large "FaceDetection",
  };


  @Test
  public void testMultiVariatelassification() throws IOException {
    try {
      // the relative path to the datasets
      ClassLoader classLoader = SFAWordsTest.class.getClassLoader();

      File dir = new File(classLoader.getResource("datasets/Multivariate_arff/").getFile());
      //File dir = new File("/Users/bzcschae/Downloads/Multivariate_arff/");

      TimeSeries.APPLY_Z_NORM = false;

      for (String s : datasets) {
        File d = new File(dir.getAbsolutePath() + "/" + s);
        if (d.exists() && d.isDirectory()) {
            File train = new File(dir.getAbsolutePath()+ "/" + s + "/" + s +"_TRAIN.arff");
            File test = new File(dir.getAbsolutePath()+ "/" + s + "/" + s +"_TEST.arff");

            if (!test.exists()) {
              System.err.println("File " + test.getName() + " does not exist");
              test = null;
            }

            Classifier.DEBUG = false;

            boolean useDerivatives = true;
            Map<String, Double> classMapping = new TreeMap<String, Double>();
            MultiVariateTimeSeries[] trainSamples
                = TimeSeriesLoader.loadMultivariateDatsetArff(train, s, classMapping, useDerivatives);
            MultiVariateTimeSeries[] testSamples
                = TimeSeriesLoader.loadMultivariateDatsetArff(test, s, classMapping, useDerivatives);

            MUSEClassifier muse = new MUSEClassifier();
            MUSEClassifier.Score museScore = muse.eval(trainSamples, testSamples);
            System.out.println(s + ";" + museScore.toString());
        } else{
          // not really an error. just a hint:
          System.out.println("Dataset could not be found: " + d.getAbsolutePath() + ".");
        }
      }
    } finally {
      TimeSeries.APPLY_Z_NORM = true; // FIXME static variable breaks some test cases!
    }
  }
}
